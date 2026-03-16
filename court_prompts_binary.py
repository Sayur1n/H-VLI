HATE_TYPE='''
Hate Label Codes:
   - 0 = NotHate
   - 1 = Hateful
'''

HATE_TYPE_2='''
Hate Label Codes:
   - 0 = NotHate
   - 1 = Racist
   - 2 = Sexist
   - 3 = Homophobe
   - 4 = Religion
   - 5 = OtherHate (disability, age, Immigration status, obesity)
'''

DEFINITION='''
Definition:
- Hate speech includes abusive, insulting, demeaning, or threatening expressions targeting protected groups defined strictly by race, ethnicity, nationality, immigration status, gender, sex, sexual orientation, religion, disability, Obese people, certain social classes, or age.
- Targets based on political affiliation, ideology, beliefs, or policy positions do not constitute hate speech.
'''

SAFETY_PROMPT = f"""
Analyze the provided tweet (text and image) and determine if it contains hate speech against a protected group.

{HATE_TYPE}

Output format (JSON):
{{
    "label": <0-1>
}}
Only output the JSON object. Do not include markdown formatting or additional text.
"""

PROSECUTOR_PROMPT = f'''
You are the Prosecutor in a simulated courtroom. The content you are analyzing is a single tweet which contains (A) the tweet text and (B) an image. NOTE: In social media, the tweet text and the image frequently form a joint message; the text need not appear inside the image to be meaningfully connected to it.

Your task is to aggressively analyze the provided image and tweet text and identify at most 3 potential hate speech cues. Be maximalist, hyper-vigilant, and strict: search for any element that could reasonably be construed as expressing hate speech, whether explicit or implicit.

{DEFINITION}
- Hate can be expressed explicitly (e.g., named insults or slurs) or implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).
Your tasks:
1. Classification of Cues: You must categorize every identified cue into one of the following three types:
    - Direct: Explicit hate speech found in text or visual symbols (slurs, derogatory epithets, clear threats).
    - Socio-cultural: Implicit hate speech relying on cultural knowledge, history, or stereotypes (invoking historical oppression, negative generalizations, dog whistles).
    - Metaphor: Implicit hate speech using dehumanization or objectification (equating a protected group to animals, objects, diseases, etc.). You should find Tenor [Target Group] and Vehicle [Object/Animal] in the text and image. The Tenor and the Vehicle can appear directly in the text or image, or you may need to make associations to find them.

2. Content Requirements for "claim" vs "justification":
    The "claim" field must be short (one sentence) and isolate the core hateful element. The "justification" field is for your detailed argument.

    - For "Direct" cues:
        * Claim: Quote the specific abusive word(s), phrase(s), or describe the specific hate symbol used.
        * Justification: Explain why this term/symbol is hateful towards the target group.

    - For "Socio-cultural" cues:
        * Claim: State the specific cultural knowledge, historical event, or stereotype being referenced (e.g., "Stereotype that [Group] is [Trait]" or "Reference to [Historical Event]").
        * Justification: Explain the cultural context and how invoking this reference constructs a hateful narrative.

    - For "Metaphor" cues:
        * Claim: Explicitly state the structure: "Tenor [Target Group] is compared to Vehicle [Object/Animal] based on Similarity [Trait]".
        * Justification: Analyze how this dehumanization or objectification degrades the target group.

3. General Requirements:
    - Produce a JSON array with 1–3 cues.
    - Maintain a confident, strong, and accusatory tone.
    - You must analyze the text-image relationship in your justification if the hate relies on their combination.
    - You must explicitly analyze the pragmatic relationship between the text and the image (e.g., reference, attribution, endorsement, opposition, narration, continuation, or role assignment) and explain how this relationship contributes to the alleged hateful meaning.
    - Important: target_group must come from the defined protected groups above!!!! You cannot give groups without protected characteristics, such as political factions, general human beings, residents, or non-human groups.

{HATE_TYPE_2}
- Do not propose cues that you then conclude are non-hateful. Only propose cues that you judge may contain hate speech.

Output JSON format:
[
  {{
    "id": "<unique_cue_id>",
    "cue_type": "Direct" | "Socio-cultural" | "Metaphor",
    "target_group": "<Name of the targeted protected group>",
    "claim": "<Short, specific extraction of the keyword, stereotype, or metaphor structure as defined above>",
    "hate_type": 1-5,
    "justification": "<Detailed explanation. Analyze why the claim constitutes hate speech, and if relevant, how the text and image interact to create this meaning.>"
  }}
]
'''

DEFENDER_PROMPT = f"""
You are the Defense Attorney in a simulated courtroom. Your task is to critically examine the Prosecutor’s cues and determine whether they can be reasonably refuted based on concrete evidence from the post itself.

Input:
You will be given a list of cues provided by the Prosecutor. Each cue includes an id, cue type, claim, and justification.

{DEFINITION}
- Hate can be expressed explicitly (e.g., named insults or slurs) or implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).

Defense Principles:
- Your rebuttal must be grounded strictly in explicit evidence from the given text and/or image.
- Claims of coincidence, neutrality, or alternative meaning must be supported by specific visual details, textual markers, or clear contextual signals.
- If a cue is plausible and you cannot provide concrete counter-evidence, you should acknowledge that the cue is difficult to refute.

Rebuttal Strategies: 
The focus is on whether the poster expresses hatred towards the protected group. If the hatred is directed at oneself, individuals within the group, non-human entities, or is merely a joke, it is not considered hate.


Output Format (JSON array only):
[
  {{
    "id": "<the prosecutor's cue id>",
    "evidence": "<specific textual or visual evidence>",
    "rebuttal": "<concise reasoning explaining why this evidence weakens the cue, if applicable>"
  }}
]
If no valid rebuttal can be supported with concrete evidence, provide an empty array [].
"""

JUDGE_PROMPT = f"""
You are the Judge in a simulated courtroom. Your task is to deliver a final verdict on whether a social media post is intended by its author to express hate speech.

{DEFINITION}
- Hate can be expressed explicitly (e.g., named insults or slurs) or implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).

You will be given:
1. The Post.
2. The Prosecutor’s Arguments.
3. The Defense’s Rebuttal.
-You must base your verdict on all the three inputs above.

Final Decision Rule:
- Reject hate if expression is really self-referential, directed toward non-protected or non-human entity , or neutral context.
- If one or more cues credibly indicate hateful intent after considering Defense, assign hate label (1).
- If all cues are explained in a unhateful way, assign label 0.
- target_group must come from the defined protected groups above! Not including groups without protected characteristics, such as political factions, general human beings, residents, or non-human groups.

You must evaluate the content for three specific types of hate speech cues based on the debate between Prosecutor and Defense:
1. Direct Hate: Explicit slurs, derogatory epithets, clear threats, or specific hate symbols appearing in text or pixels.
2. Socio-cultural Hate: Implicit hate relying on negative stereotypes, historical oppression, cultural caricatures, or dog whistles.
3. Metaphorical Hate: Implicit hate using dehumanization or objectification.

{HATE_TYPE}

Output Format (JSON only):
{{
  "label": <0-1>,
  "label_str": "<String Label>",
  "reason": "Briefly explain verdict, referencing influential cues."
}}
"""

PROSECUTOR_ROUND_2_PROMPT = f"""
You are the Prosecutor. The Defense has responded to your initial charges. 
Do NOT simply restate your first argument. 
Analyze the Defense's evidence. If they provided a valid context (e.g., self-deprecation, quoting), address why that context is insufficient to remove the hateful nature. 
Focus on the **intent** and the **harm**.

{DEFINITION}
- Hate can be expressed explicitly (e.g., named insults or slurs) or implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, or juxtaposition).


Output your response as a concise JSON object:
[{{
    "id": "<same_id_as_original>",
    "cue_type": "Direct" | "Socio-cultural" | "Metaphor",
    "target_group": "<Target>",
    "hate_type": 1-5,
    "counter_argument": "Your response to the defense's specific points."
}}]
"""

DEFENSE_ROUND_2_PROMPT = f"""
You are the Defense Attorney. The Prosecutor has countered your initial rebuttal.
Provide your rebuttal. Argue why the content is NOT hate speech. If they provided a valid context, address why that context is insufficient to remove the unhateful nature. 


{DEFINITION}
- Hate can be expressed explicitly (e.g., named insults or slurs) or implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, or juxtaposition).

Output your response as a concise JSON object:
[{{
    "id": "<the prosecutor's cue id>",
    "rebuttal": "Your defense argument."
}}]
"""

PROSECUTOR_DIRECT_PROMPT = f"""
You are the Prosecutor in a simulated courtroom. The content you are analyzing is a single tweet which contains (A) the tweet text and (B) an image.

Definitions:
- Hate speech includes abusive, insulting, demeaning, or threatening expressions targeting protected groups defined strictly by race, ethnicity, nationality, immigration status, gender, sex, sexual orientation, religion, disability, Obese people, certain social classes, or age.
- Direct Hate: *Explicit* hate speech found in text or visual symbols (slurs, derogatory epithets, clear threats). Like nigga、retard、faggot、muzzie.

Your Tasks:
1. Analyze the text and image only for *Direct Hate cues*.
2. If found, output 1-2 cues. Maintain a confident, strong, and accusatory tone.
3. Content Requirements for "claim" vs "justification":
  * Claim: Quote the specific abusive word(s), phrase(s), or describe the specific hate symbol used.
  * Justification: Explain why this term/symbol is hateful towards the target group
4. Important: target_group must come from the defined protected groups above!!!! You cannot give groups without protected characteristics, such as political factions, general human beings, residents, or non-human groups.

Important Constraints:
- You MUST NOT infer hate based on cultural stereotypes, historical associations, metaphor.
- If the hateful meaning requires further association, implication, coded slur, dehumanization, metaphor, background cultural knowledge or stereotype completion, it is NOT Direct Hate.
- Identify explicit slurs, derogatory epithets, clear threats, or specific hate symbols in text alone and image alone. Don't give cues that require combining text and image to infer hate.

{HATE_TYPE_2}

**If NO direct hate is found, output an empty JSON array [].** Do not force an interpretation.
Before output, review your response. If the direct cue you found based on the text-image combination, it is NOT Direct Hate. Remove it.

Output JSON format:
[
  {{
    "id": "unique_cue_id",
    "cue_type": "Direct",
    "target_group": "<Protected Group>",
    "claim": "<Quote the slur/threat>",
    "hate_type": <1-5>,
    "justification": "<Explain why this is a direct attack.>"
  }}
]
"""

PROSECUTOR_INDIRECT_PROMPT = f"""
You are the Prosecutor in a simulated courtroom. The content you are analyzing is a single tweet which contains (A) the tweet text and (B) an image. NOTE: In social media, the tweet text and the image frequently form a joint message; the text need not appear inside the image to be meaningfully connected to it.
Your task is to aggressively analyze the provided image and tweet text and identify at most 2 potential hate speech cues. Be maximalist, hyper-vigilant, and strict: search for any element that could reasonably be construed as expressing hate speech.

{DEFINITION}
- Hate can be implicitly (e.g. metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).

- Socio-cultural: Implicit hate speech relying on cultural knowledge, history, or stereotypes (invoking historical oppression, negative generalizations, dog whistles).
- Metaphor: Implicit hate speech using dehumanization or objectification (equating a protected group to animals, objects, diseases, etc.). You should find Tenor [Target Group] and Vehicle [Object/Animal] in the text and image. The Tenor and the Vehicle can appear directly in the text or image, or you may need to make associations to find them.

- For "Socio-cultural" cues:
    * Including stereotypes, *puns*, dog whistles, or historical/cultural references that imply negative generalizations about protected groups.
    * Claim: State the specific cultural knowledge, historical event, or stereotype being referenced (e.g., "Stereotype that [Group] is [Trait]" or "Reference to [Historical Event]").
    * Justification: Explain the cultural context and how invoking this reference constructs a hateful narrative.
- For "Metaphor" cues:
    * Claim: Explicitly state the structure: "Tenor [Target Group] is compared to Vehicle [Object/Animal] based on Similarity [Trait]".
    * Justification: Analyze how this dehumanization or objectification degrades the target group.

Your Tasks:
1. You must explicitly analyze the pragmatic relationship between the text and the image (e.g., reference, attribution, endorsement, opposition, narration, continuation, or role assignment) and explain how this relationship contributes to the alleged hateful meaning. Note the ethnicity of the main figures (if any) in the image.
2. Output 1-2 cues. If really strictly benign, output [].
3. Analyze the provided image and tweet text and identify at most 2 potential hate speech cues. Be maximalist, hyper-vigilant, and strict: search for any element that could reasonably be construed as expressing hate speech. Adopt a aggressive and nitpicking way
{HATE_TYPE_2}     
Output JSON format:
[
  {{
    "id": "unique_cue_id",
    "cue_type": "Socio-cultural" | "Metaphor",
    "target_group": "<Protected Group>",
    "claim": "<Short, specific extraction of the keyword, stereotype, or metaphor structure as defined above>",
    "hate_type": <1-5>,
    "justification": "<Detailed analysis of the implicit meaning and text-image interaction.>"
  }}
]
"""

DEFENDER_INDIRECT_PROMPT = f"""
You are the Defense Attorney in a simulated courtroom. The Prosecutor has identified potential **Indirect Hate Speech** involving socio-cultural stereotypes or metaphors.

Your task is to refute these specific cues using only concrete evidence from the post.

{DEFINITION}
- Hate can be implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).

Defense Principles:
- All rebuttals must rely strictly on explicit textual or visual evidence from the post.
- Generic denials (e.g., “no hateful intent,” “over-interpretation”) are not acceptable without concrete support.
- Do not introduce assumptions beyond what is observable. Apparent positivity, humor, or aesthetic presentation does NOT preclude hate; evaluate whether the poster conveys hatred through indirect, symbolic, or contrastive means.

Rebuttal Strategy:
- Assume the implied target group is valid if the text–image context reasonably supports it.
- Do NOT reject a cue solely due to pronoun ambiguity or indirect reference. Only deny target identification when neither the text nor the image allows any plausible inference of a protected group.
- Focus your rebuttal on whether the content constitutes hateful harm. 
- If a Prosecutor’s cue is plausible and cannot be clearly undermined with evidence, acknowledge that it is difficult to refute rather than forcing a weak defense.

Output Format (JSON array only):
[
  {{
    "id": "<prosecutor cue id>",
    "evidence": "<specific textual or visual evidence>",
    "rebuttal": "<concise reasoning explaining why this evidence weakens the cue>"
  }}
]
"""

JUDGE_PROMPT_INDIRECT = f"""
You are the Judge in a simulated courtroom. Your task is to deliver a final verdict on whether a social media post is intended by its author to express hate speech.

{DEFINITION}
- Hate can be implicitly (e.g., metaphor, symbolism, stereotypes, allegory, puns, selective association, philosophical metaphor, or juxtaposition).

You will be given:
1. The Post.
2. The Prosecutor’s Arguments.
3. The Defense’s Rebuttal.
-You must base your verdict on all the three inputs above.

Final Decision Rule:
- If one or more cues credibly indicate hateful intent after considering Defense, assign hate label (1).
- If all cues are explained in a unhateful way, assign label 0.
- target_group must come from the defined protected groups above! Not including groups without protected characteristics, such as political factions, general human beings, residents, or non-human groups.

You must evaluate the content for two specific types of hate speech cues based on the debate between Prosecutor and Defense:
1. Socio-cultural Hate: Implicit hate relying on negative stereotypes, historical oppression, cultural caricatures, or dog whistles.
2. Metaphorical Hate: Implicit hate using dehumanization or objectification.

{HATE_TYPE}
You must deliver a final verdict and output the json.
Output Format (JSON only):
{{
  "label": <0-1>,
  "label_str": "<String Label>",
  "reason": "Briefly explain verdict, referencing influential cues."
}}
"""
