import copy
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(timeout=1500)

ds = load_dataset("google/simpleqa-verified")

problems = []
for i in range(len(ds["eval"])):
    problem = ds["eval"][i]["problem"]
    original_index = ds["eval"][i]["original_index"]
    id = original_index
    problems.append({"id":id, "problem":problem})

class AnswerFormat(BaseModel):
    explanation: str
    answer: str


force_percentages_list = ["10", "25", "50"]
selective_percentages_list = ["50"]
# languages_list = ["Klingon","Dothraki","Afrikaans","Albanian","Amharic","Arabic","Armenian","Azerbaijani","Burmese","Danish","Dutch","Finnish","French","Georgian","German","Greek","Hebrew","Hungarian","Icelandic","Javanese","Kannada","Khmer","Korean","Latvian","Malay","Malayalam","Mongolian","Norwegian","Persian","Polish","Portuguese","Romanian","Russian","Slovenian","Sundanese","Swahili","Traditional Chinese","Swedish","Tagalog","Tamil","Telugu","Thai","Turkish","Vietnamese","Welsh","Japanese","Indonesian","Italian","Simplified Chinese","Hindi","Marathi","Spanish"]
# languages_list = ["Indonesian","Italian"]
# languages_list = ["Simplified Chinese","Urdu","Traditional Chinese"]
languages_list = ["Toba Batak"]
# languages_list = ["Hindi", "Marathi","Spanish"]

NUM_SPLIT = 20
SAMPLES_PER_SPLIT = len(problems) // NUM_SPLIT

for split in range(10, NUM_SPLIT):
    min_id = split * SAMPLES_PER_SPLIT
    max_id = (split+1) * SAMPLES_PER_SPLIT
    if split == NUM_SPLIT-1:
        max_id = len(problems)

    for language_list in languages_list:
        if type(language_list) is list:
            languages = "_".join(lang for lang in language_list)
        else:
            languages = language_list
            language_list = [language_list]

        os.system(f"mkdir -p \"output/{languages}/\"")

        print(languages, "from", languages_list, "split", split, "min_id", min_id, "max_id", max_id)
        outputs = {}
        outputs[languages] = {}
        for i in tqdm(range(min_id, max_id)):
            all_responses = {"original": problems[i]["problem"]}
            text = problems[i]["problem"]

            for percentage in force_percentages_list:
                all_responses[percentage + "_force"] = []
                all_responses[percentage + "_force_backtranslate"] = []

                for j in range(1):
                    completion = client.chat.completions.parse(
                        model="gpt-5.2",
                        reasoning_effort="low",
                        messages=[
                            {"role": "developer", "content": "You are a multilingual speaker."},
                            {"role": "user", "content": f"""Given an English text, produce a code-switched version with roughly about {percentage}% of the words or phrases into {languages}, while preserving the original meaning.\nApply code-switching by force. Don't write new punctuation if not needed. The answer must not have any preamble.\nEnglish text: {text}"""}
                        ],
                        service_tier="flex",
                        timeout=1500,
                        response_format=AnswerFormat
                    )
                    
                    output = completion.choices[0].message.content
                    all_responses[percentage + "_force"].append(output)

                    completion = client.chat.completions.parse(
                        model="gpt-5.2",
                        reasoning_effort="low",
                        messages=[
                            {"role": "developer", "content": "You are a multilingual speaker."},
                            {"role": "user", "content": f"""Translate the text to English. Output only the final translation text.\nText: {output}"""}
                        ],
                        service_tier="flex",
                        timeout=1500,
                        response_format=AnswerFormat
                    )
                    output = completion.choices[0].message.content
                    all_responses[percentage + "_force_backtranslate"].append(output)

            for percentage in selective_percentages_list:
                all_responses[percentage + "_selective"] = []
                for lang in language_list:
                    all_responses["translate_" + lang] = []
                    all_responses[percentage + "_grammarforce_" + lang] = []
                    all_responses[percentage + "_backtranslate_grammarforce_" + lang] = []
                all_responses[percentage + "_grammarforce_English"] = []
                all_responses[percentage + "_backtranslate_grammarforce_English"] = []
                all_responses[percentage + "_backtranslate"] = []

                for j in range(1):
                    completion = client.chat.completions.parse(
                        model="gpt-5.2",
                        reasoning_effort="low",
                        messages=[
                            {"role": "developer", "content": "You are a multilingual speaker."},
                            {"role": "user", "content": f"""Given an English text, produce a code-switched version with roughly about {percentage}% of the words or phrases into {languages}, while preserving the original meaning.\nApply code-switching selectively, BUT always try to code-switch if possible, so that the final output naturally mixes English with the target language(s). Don't write new punctuation if not needed. The answer must not have any preamble.\nEnglish text: {text}"""}
                        ],
                        service_tier="flex",
                        timeout=1500,
                        response_format=AnswerFormat
                    )
                    
                    output = completion.choices[0].message.content
                    all_responses[percentage + "_selective"].append(output)

                    completion = client.chat.completions.parse(
                        model="gpt-5.2",
                        reasoning_effort="low",
                        messages=[
                            {"role": "developer", "content": "You are a multilingual speaker."},
                            {"role": "user", "content": f"""Translate the text to English. Output only the final translation text.\nText: {output}"""}
                        ],
                        service_tier="flex",
                        timeout=1500,
                        response_format=AnswerFormat
                    )
                    output = completion.choices[0].message.content
                    all_responses[percentage + "_backtranslate"].append(output)

                    for lang_id in range(len(language_list)):
                        lang = language_list[lang_id]

                        for gf_lang in language_list + ["English"]:
                            completion = client.chat.completions.parse(
                                model="gpt-5.2",
                                reasoning_effort="low",
                                messages=[
                                    {"role": "developer", "content": "You are a multilingual speaker."},
                                    {"role": "user", "content": f"""Given an English text, produce a code-switched version with roughly about {percentage}% of the words or phrases into {lang}, while preserving the original meaning.\nApply code-switching selectively, BUT always try to code-switch if possible, so that the final output naturally mixes English with the target language(s) and force the code-switched text to follow {gf_lang} grammar. Don't write new punctuation if not needed. The answer must not have any preamble.\nEnglish text: {text}"""}
                                ],
                                service_tier="flex",
                                timeout=1500,
                                response_format=AnswerFormat
                            )
                            
                            output = completion.choices[0].message.content
                            all_responses[percentage + "_grammarforce_" + gf_lang].append(output)

                            completion = client.chat.completions.parse(
                                model="gpt-5.2",
                                reasoning_effort="low",
                                messages=[
                                    {"role": "developer", "content": "You are a multilingual speaker."},
                                    {"role": "user", "content": f"""Translate the text to English. Output only the final translation text.\nText: {output}"""}
                                ],
                                service_tier="flex",
                                timeout=1500,
                                response_format=AnswerFormat
                            )
                            output = completion.choices[0].message.content
                            all_responses[percentage + "_backtranslate_grammarforce_" + gf_lang].append(output)

                    for lang_id in range(len(language_list)):
                        lang = language_list[lang_id]
                        completion = client.chat.completions.parse(
                            model="gpt-5.2",
                            reasoning_effort="low",
                            messages=[
                                {"role": "developer", "content": "You are a multilingual speaker."},
                                {"role": "user", "content": f"""Translate the text to {lang}. Output only the final translation text.\nText: {text}"""}
                            ],
                            service_tier="flex",
                            timeout=1500,
                            response_format=AnswerFormat
                        )
                        output = completion.choices[0].message.content
                        all_responses["translate_" + lang].append(output)

            outputs[languages][i] = all_responses

        with open(f'output/{languages}/{languages}_split_{split}.json', 'w', encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)