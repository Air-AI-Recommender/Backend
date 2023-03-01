from fastapi import FastAPI, Query
from typing import List, Dict
from pydantic import BaseModel, validator
import openai
import json
import requests
import re
import time
import tiktoken
import pandas as pd
from fuzzywuzzy import fuzz
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

SECRETS = json.load(open("OPENAI_KEY.json"))
openai.api_key = SECRETS["OPENAI_KEY"]
MODEL_ENGINE = "text-davinci-003"
PATTERN = r'(\d+)\.(.+)\s\(([\d]+)\)'
# PATTERN_EXPLANATION = r"(\d+)\. ([^\n]+) \((\d{4})\): ([^\n]+)"
PATTERN_EXPLANATION = r'(\d+)\.\s(.*?):\s(.*)'
REASONS = {}

SERVICES = ['netflix', 'prime', 'disney', 'hbo', 'hulu', 'peacock', 'paramount', 'starz', 'showtime', 'apple', 'mubi']
DF = pd.DataFrame.from_dict(json.load(open('all_media_data.json', 'r')), orient='index').fillna('NaN')
GENRES = list(requests.request("GET", "https://streaming-availability.p.rapidapi.com/genres", headers={
	"X-RapidAPI-Key": "38c146f2c9msh90d9985ae589b34p15eb28jsn87c134334533",
	"X-RapidAPI-Host": "streaming-availability.p.rapidapi.com"
}).json().values())
CACHE = {}

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

app = FastAPI()

class Media(BaseModel):
    title: str = Query("The Matrix")
    reason: str = Query(
        "The action!", description="The (optional) user reason for their rating"
    )

def get_results(text, df, ignore):
    results = []
    matches = re.findall(PATTERN, text)
    seen = set()
    for match in matches:
        num, title, year = match
        title = title.strip()
        if title not in seen:
            results.append({"title": title})
            seen.add(title)
        
    idx_del = []
    for i, result in enumerate(results):
        if result['title'] in ignore:
            idx_del.append(i)
            continue
        
        t = df[df['title'].apply(lambda x: fuzz.ratio(x.lower(), result['title'].lower()) > 90)]
        if t.shape[0] == 0:
            idx_del.append(i)
        else:
            result['streaming_data'] = t.iloc[0].to_dict()
         
    ignored = []
    for ele in sorted(idx_del, reverse = True):
        ignored.append(results[ele]['title'])
        del results[ele]
        
    return results, ignored

def call_openai(prompt):
    num_tokens = num_tokens_from_string(prompt, "gpt2")
    max_tokens = 4000 - num_tokens - 20
    logging.info(f"------NUM_TOKENS-------{num_tokens},{max_tokens}")
    # Set the parameters for the API call
    params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1,
        "top_p": 1,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    worked = False
    while not worked:
        try:
            response = openai.Completion.create(
                engine=MODEL_ENGINE,
                **params
            )
            worked = True
        except:
            time.sleep(10)
    
    text = response.choices[0]["text"]
    return text

@app.post("/recommendations/")
def get_recommendations(
        preferences: list[Media], 
        services: list = SERVICES, 
        genres: list = GENRES, 
        content_type: str = Query("all", enum=['movies', 'series']),
        ignore: list = [],
    ):
    logging.info(f"------GET_RECOMMENDATIONS-------")
    preferences = [preference.dict() for preference in preferences]
    info_hash = hash(tuple([item for sublist in [p.items() for p in preferences] for item in sublist]) + tuple(services) + tuple(genres) + tuple([content_type]) + tuple(ignore))
    logging.info(f"------INFO_HASH-------: {info_hash}")
    if info_hash in CACHE:
        return CACHE[info_hash]
    
    user_preferences = ""
    if len(genres) <= 5:
        user_preferences = f"User is only interested in the following genres: {','.join(genres)}"
        
    user_preferences += " The user liked the following:"
    for preference in preferences:
        reason = f" ({preference['reason']})" if preference['reason'] else ""
        user_preferences += f" {preference['title']}{reason},"

    content = ""
    if content_type == 'all':
        content = "TV shows and movies"
    if content_type == 'movies':
        content = "movies"
    if content_type == 'series':
        content = "TV shows"

    prompt = f"user_preferences: {user_preferences}\n\n"
    prompt += f"Recommend {max(25, int(100/len(SERVICES)))} {content} based on the users preferences."
    prompt += " Include the title as it would appear on IMDB."
    if len(ignore) > 0:
        prompt += f" Do not recommend the following as they have already been recommended: {','.join(list(ignore))}."
    prompt += f" Output in a format that works with this regex: {PATTERN}"
    logging.info(f"------PROMPT-------: {prompt}")

    text = call_openai(prompt)
    logging.info(f"------OPENAI_RESPONSE-------: {text}")
        
    df = DF.copy()
    if services != SERVICES:
        df = df[df['streamingInfo'].apply(lambda x: any([k in services for k in x.keys()]))]
        
    ignore += [preference['title'] for preference in preferences]
    results, ignored = get_results(text, df, ignore)
    logging.info(f"------RESULTS_LEN-------: {len(results)}")
    
    prompt = f"user_preferences: {user_preferences}\n\n"
    prompt += " Give a long and detailed explanation on the reasoning behind why the user would like each of the following TV shows and/or movies based on their preferences:"
    prompt += f" {','.join([r['title'] for r in results])}"
    prompt += f" Output in a format that works with this regex: {PATTERN_EXPLANATION}"
    logging.info(f"------PROMPT_EXPLANATION-------: {prompt}")
    
    matches = []
    attempts = 0
    while len(matches) != len(results) and attempts < 3:
        text = call_openai(prompt)
        logging.info(f"------OPENAI_RESPONSE_EXPLANATION_{attempts}-------: {text}")
        matches = re.findall(PATTERN_EXPLANATION, text)
        attempts += 1
    
    for i, match in enumerate(matches):
        num, title, explanation = match
        results[i].update({"explanation": explanation})
        
    CACHE[info_hash] = {"prompt": prompt, "response": text, "recommendations": results, "ignored": ignored}
    return CACHE[info_hash]

@app.post("/content/")
def get_content(
        services: list = SERVICES, 
        genres: list = GENRES, 
        content_type: str = Query("all", enum=['movies', 'series']), 
        min_imdb_rating: int = Query(None, example=70),
        sort_by: str = Query("imdb_rating", enum=['imdb_rating', 'alphabetically', 'release_date']),
        sort_by_reversed: bool = False,
        page: int = 1, 
    ):
    logging.info(f"------GET_CONTENT-------")
    df = DF.copy()
    if content_type != 'all':
        df = df[df['content_type'] == content_type]
    if services != SERVICES:
        df = df[df['streamingInfo'].apply(lambda x: any([k in services for k in x.keys()]))]
    if genres != GENRES:
        df = df[df['genres'].apply(lambda x: any([k in genres for k in x]))]
    if min_imdb_rating:
        df = df[df['imdbRating'].apply(lambda x: x > min_imdb_rating)]
        
    if sort_by == 'imdb_rating':
        df = df[df['imdbVoteCount'] > 1000].sort_values('imdbRating', ascending=False if not sort_by_reversed else True)
    elif sort_by == 'alphabetically':
        df = df.sort_values('title', ascending=True if not sort_by_reversed else False)
    elif sort_by == 'release_date':
        df = df.sort_values('year', ascending=False if not sort_by_reversed else True)
    
    return df.iloc[page*50:(page+1)*50].to_dict(orient='records')

@app.post("/reasons/")
def get_reasons(titles: list[str] = []):
    logging.info(f"------GET_REASONS-------")
    pattern = r"(\d+)\.(.*)"

    for title in titles:
        if title in REASONS:
            continue
        
        prompt = f"Give 5 completely different reasons why someone might like the following TV shows and/or movies: {title}. Make them one sentence."
        prompt += f" Don't include the title. Output in a format that works with this regex: {pattern}"
        logging.info(f"------PROMPT_REASONS_{title}-------: {prompt}")
        
        text = call_openai(prompt)
        logging.info(f"------OPENAI_RESPONSE_REASONS_{title}-------: {text}")

        matches = re.findall(pattern, text)

        t = []
        for match in matches:
            num, reason = match
            t.append(reason.strip())
            
        REASONS[title] = t
        
    return {title: REASONS[title] for title in titles}

@app.get("/search")
def search(movie_title: str):
    df = DF[DF['title'].str.lower().str.contains(movie_title)].copy()
    return df.to_dict(orient='records')