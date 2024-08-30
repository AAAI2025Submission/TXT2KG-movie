import pandas as pd
import ast
import os
import re


def sanitize_filename(filename: str) -> str:
    # Define a set of characters that are not allowed in file names
    illegal_chars = r'[<>:"/\\|?*]'

    # Replace illegal characters with an underscore
    sanitized_filename = re.sub(illegal_chars, '_', filename)

    # Optionally, strip leading and trailing whitespace
    sanitized_filename = sanitized_filename.strip()

    return sanitized_filename

# with open('data/movies.xlsx', 'rb') as f:
#     df = pd.read_excel(f).fillna("")
# with open('data/movies2.csv', 'r',encoding='utf-8') as f2:
#     df2=pd.read_csv(f2).fillna("")
# if not os.path.exists('movies'):
#     os.mkdir('movies')
# for i in range(len(df)):
#     id=df.loc[i,'imdb']
#     idx2=df2[df2['IMDb']==id].index[0]
#
#     movie_name=df.loc[i,'movie_name']
#     director=df.loc[i,'director']
#     writers=df.loc[i,'writers']
#     all_actors=df.loc[i,'all_actors']
#     main_actors=df.loc[i,'top5_actors']
#     producers=df.loc[i,'producers']
#     music=df.loc[i,'music']
#
#     descriptions=df2.loc[idx2,'Description']
#     genres="; ".join(ast.literal_eval(df2.loc[idx2,'Genre']))
#     district="; ".join(ast.literal_eval(df2.loc[idx2,'District']))
#     language="; ".join(ast.literal_eval(df2.loc[idx2,'Language']))
#     year=df2.loc[idx2,'Year']
#     if int(year)<=2014:
#         continue
#     movie_filename = sanitize_filename(movie_name)
#     with open(f'movies/{movie_filename}.txt','w',encoding='utf-8') as f:
#         f.write(f'Movie name: {movie_name}\n')
#         f.write(f'Director: {director}\n')
#         f.write(f'Genres: {genres}\n')
#         f.write(f'District: {district}\n')
#         f.write(f'Description: {descriptions}\n')
#         f.write('\n')
#         f.write(f'Writers:\n{writers}\n')
#         f.write('\n')
#         f.write(f'Main Actors:\n{main_actors}\n')
#         f.write('\n')
#         f.write(f'Producers:\n{producers}\n')
#         f.write('\n')
#         f.write(f'ALL Actors:\n{all_actors}\n')
#         f.write('\n')
#         f.write(f'Music:\n{music}\n')
#         f.write('\n')

with open('data/people.xlsx', 'rb') as f:
    df3 = pd.read_excel(f).fillna("")

if not os.path.exists('people'):
    os.mkdir('people')
for i in range(len(df3)):
    name=df3.loc[i,'ppl_name']
    person_filename = sanitize_filename(name)

    description1=df3.loc[i,'info1']
    description2=df3.loc[i,'info2']
    description3=df3.loc[i,'info3']
    description4=df3.loc[i,'info4']
    description5=df3.loc[i,'info5']

    with open(f'people/{person_filename}.txt','w',encoding='utf-8') as f:
        f.write(f'Name: {name}\n')
        f.write('\n')
        f.write(f'{description1}\n')
        f.write(f'{description2}\n')
        f.write(f'{description3}\n')
        f.write(f'{description4}\n')
        f.write(f'{description5}\n')