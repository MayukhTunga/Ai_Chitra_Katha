import os
import replicate
import io
import warnings
from IPython.display import display
from PIL import Image
from stability_sdk import client
from stability_sdk.client import generation
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog


from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
load_dotenv(find_dotenv())  

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

text = image_to_text("B:\Ai_chitra_khata\download.jpeg")[0]['generated_text']


promptvar=text

promptvar = "Write a long story from this: "+promptvar

os.environ["Replicate_API_Token"] =""

output = replicate.run(
    "meta/llama-2-70b-chat:",
    input={
        "prompt": promptvar
    }
)

story="Summarize the following in 5 points:  "
# The meta/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/meta/llama-2-70b-chat/api#output-schema
    #print(item, end="")
    story += item
print(story)

output = replicate.run(
    "meta/llama-2-70b-chat:",
    input={
        "prompt": story
    }
)

# The meta/llama-2-70b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
story=""
for item in output:
    # https://replicate.com/meta/llama-2-70b-chat/api#output-schema
    #print(item, end="")
    story+=item

print(story)

pt = ["1.","2.","3.","4.","5."]
pr=[0,1,2,3,4]
for i in range(0,5):
    try:
        pr[i] = story.index(str(pt[i]))
    except:
        pr[i]=99
    else:
        continue

temp=""
for i in range (0,5):
    try:
        if(pr[i]!=99 and pr[i+1]!=99):
            temp=story[pr[i]+2:pr[i+1]]
            if(len(temp)>10):
                pt[i]=temp
            else:
                continue
        elif(pr[i]!=99 and pr[i]==99):
            temp = story[pr[i]+2:-1]
            if(len(temp)>10):
                pt[i]=temp
            else:
                continue
        else:
            continue
    except:
        break

stability_api = client.StabilityInference(
    host="grpc.stability.ai:443",
    key="",
    verbose=True,
)

name = "image"

for i in range(0,len(pt)):
    answers = stability_api.generate(
    prompt="Generate in black and white for: "+pt[i],
    seed=121245125, # if provided, specifying a random seed makes results deterministic
    steps=40, # defaults to 30 if not specified
    )

# iterating over the generator produces the api response
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                name = "name"+str(i+1)
                img.save(name + ".png")