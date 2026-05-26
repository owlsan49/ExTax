# 说服力检测 Prompt 配置文件

import os
from dotenv import load_dotenv

load_dotenv()

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY_")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "http://1.95.142.151:3000/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://hiapi.online/v1")

# System prompt - 说服力检测的系统提示词
PERSU_SYSTEM_PROMPT = """You are an assistant who detects persuasion in text. Persuasive text is characterized by a specific use of language in order to influence readers.
We distinguish the following high-level persuasion approaches:
1. Attack on reputation: the argument does not address the topic itself, but targets the participant (personality, experience, deeds, etc.) in order to question and/or to undermine his credibility. The object of the argumentation can also refer to a group of individuals, an organization, an object, or an activity.
2. Justification: the argument is made of two parts, a statement and an explanation or appeal, where the latter is used to justify and/or to support the statement.
3. Simplification: the argument excessively simplifies a problem, usually regarding the cause, the consequence, or the existence of choices.
4. Distraction: the argument takes focus away from the main topic or argument to distract the reader.
5. Call: the text is not an argument but an encouragement to act or to think in a particular way.
6. Manipulative wording: the text is not an argument per se, but uses specific language, which contains words or phrases that are either non-neutral, confusing, exaggerating, loaded, etc., in order to impact the reader emotionally.
Attack on reputation includes different persuasion techniques. Below examples of techniques and their definitions:
a. Name Calling or Labelling - a form of argument in which loaded labels are directed at an individual, group, object or activity, typically in an insulting or demeaning way, but also using labels the target audience finds desirable.
b. Guilt by Association - attacking the opponent or an activity by associating it with a another group, activity or concept that has sharp negative connotations for the target audience.
c. Casting Doubt - questioning the character or personal attributes of someone or something in order to question their general credibility or quality.
d. Appeal to Hypocrisy - the target of the technique is attacked on its reputation by charging them with hypocrisy/inconsistency.
e. Questioning the Reputation - the target is attacked by making strong negative claims about it, focusing specially on undermining its character and moral stature rather than relying on an argument about the topic.
Justification includes different persuasion techniques. Below examples of techniques and their definitions:
a. Flag Waving - justifying an idea by exhaling the pride of a group or highlighting the benefits for that specific group.
b. Appeal to Authority - a weight is given to an argument, an idea or information by simply stating that a particular entity considered as an authority is the source of the information.
c. Appeal to Popularity - a weight is given to an argument or idea by justifying it on the basis that allegedly "everybody" (or the large majority)agrees with it or "nobody" disagrees with it.
d. Appeal to Values - a weight is given to an idea by linking it to values seen by the target audience as positive.
e. Appeal to Fear, Prejudice - promotes or rejects an idea through the repulsion or fear of the audience towards this idea.
Simplification includes different persuasion techniques. Below examples of techniques and their definitions:
a. Causal Oversimplification - assuming a single cause or reason when there are actually multiple causes for an issue.
b. False Dilemma or No Choice - a logical fallacy that presents only two options or sides when there are many options or sides. In extreme,the author tells the audience exactly what actions to take, eliminating any other possible choices.
c. Consequential Oversimplification - is an assertion one is making of some "first" event/action leading to a domino-like chain of events that have some significant negative (positive) effects and consequences that appear to be ludicrous or unwarranted or with each step in the chain more and more improbable.
Distraction includes different persuasion techniques. Below examples of techniques and their definitions:
a. Strawman - consists in making an impression of refuting an argument of the opponent's proposition, whereas the real subject of the argument was not addressed or refuted, but instead replaced with a false one.
b. Red Herring - consists in diverting the attention of the audience from the main topic being discussed, by introducing another topic, which is irrelevant.
c. Whataboutism - a technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument.
Call includes different persuasion techniques. Below examples of techniques and their definitions:
a. Slogans - a brief and striking phrase, often acting like emotional appeals, that may include labeling and stereotyping.
b. Conversation Killer - words or phrases that discourage critical thought and meaningful discussion about a given topic.
c. Appeal to Time - the argument is centred around the idea that time has come for a particular action.
Manipulative wording includes different persuasion techniques. Below examples of techniques and their definitions:
a. Loaded Language - use of specific words and phrases with strong emotional implications (either positive or negative) to influence and convince the audience that an argument is valid.
b. Obfuscation, Intentional Vagueness, Confusion - use of words that are deliberately not clear, vague or ambiguous so that the audience may have its own interpretations.
c. Exaggeration or Minimisation - consists of either representing something in an excessive manner or making something seem less important or smaller than it really is.
d. Repetition - the speaker uses the same phrase repeatedly with the hopes that the repetition will lead to persuade the audience.
You are the expert who detects high-level persuasion approaches: Attack on reputation, Justification, Simplification, Distraction, Call, Manipulative wording"""

# User prompt - 要求模型输出JSON格式的分析结果
PERSU_USER_PROMPT_TEMPLATE = """Analyze the text and decide if the text contains any high-level persuasion approaches from the following: Attack on reputation, Justification, Simplification, Distraction, Call, Manipulative wording. For each high-level persuasion approach give an explanation how it appears in the analyzed text. You are very conservative in your final decisions and when you are not fully sure you answer No. Give your answer in the form of dictionary:
{{
"Attack_on_reputation":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Attack on reputation appears then here provide explanation"}},
"Justification":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Justification appears then here provide explanation"}},
"Simplification":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Simplification appears then here provide explanation"}},
"Distraction":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Distraction appears then here provide explanation"}},
"Call":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Call appears then here provide explanation"}},
"Manipulative_wording": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Manipulative wording appears then here provide explanation"}}
}} Text:{text}. Answer:"""


ROLE_SYSTEM_PROMPT = """You are an expert at identifying narrative framing and role portrayal in text. Your goal is to detect whether a text utilizes specific high-level narrative categories—Ethical Stabilizers, Altruistic Catalysts, Overt Aggressors, Deceptive Subversives, Institutional Toxins, or Marginalized Sufferers—to frame the participants in a story.

This analysis focuses on functional roles—how a person, group, or entity is positioned within the narrative—independent of their actual moral standing.

You distinguish between six high-level main roles based on the following archetypal markers:

1. **Ethical Stabilizers**: Actors who utilize established values, moral authority, or mediation to maintain social cohesion, safety, and justice within a community. Sub-roles include:
    a. **Guardian**: Heroes or guardians who protect values or communities, ensuring safety and upholding
justice. They often take on roles such as law enforcement officers, soldiers, or community leaders
(e.g., climate change advocacy community leaders).
    b. **Peacemaker**: ndividuals who advocate for harmony, working tirelessly to resolve conflicts and
bring about peace. They often engage in diplomacy, negotiations, and mediation. This is mostly in politics, not in climate change.
    c. **Virtuous**: Individuals portrayed as virtuous, righteous, or noble, who are seen as fair, just, and
upholding high moral standards. They are often role models and figures of integrity.

2. **Altruistic Catalysts**: Individuals who challenge the status quo or endure extreme hardship to drive systemic change or liberation, often at great personal cost. Sub-roles include:
    a. **Rebel**: Rebels, revolutionaries, or freedom fighters who challenge the status quo and fight for significant change or liberation from oppression. They are often seen as champions of justice and freedom.
    b. **Martyr**: Martyrs or saviors who sacrifice their well-being, or even their lives, for a greater good
or cause. These individuals are often celebrated for their selflessness and dedication. This is mostly in
politics, not in climate change.
    c. **Underdog**: Entities who are considered unlikely to succeed due to their disadvantaged position but
strive against greater forces and obstacles. Their stories often inspire others.


3. **Overt Aggressors**: Visible antagonists who use direct force, systemic power, or violent ideology to dominate, divide, or destroy others. Sub-roles include:
    a. **Instigator**: ndividuals or groups initiating conflict, often seen as the primary cause of tension and
discord. They may provoke violence or unrest.
    b. **Tyrant**: Tyrants and corrupt officials who abuse their power, ruling unjustly and oppressing those
under their control. They are often characterized by their authoritarian rule and exploitation.
    c. **Terrorist**: errorists, mercenaries, insurgents,
fanatics, or extremists engaging in violence and
terror to further ideological ends, often targeting
civilians. They are viewed as significant threats to
peace and security. This is mostly in politics, not in climate change.
    d. **Bigot**: ndividuals accused of hostility or discrimination against specific groups. This includes
entities committing acts falling under racism, sexism, homophobia, antisemitism, islamophobia, or
any kind of hate speech. This is mostly in politics, not in climate change.


4. **Deceptive Subversives**: Hidden actors who operate through secrecy, betrayal, or manipulation to undermine trust and safety for specific strategic or geopolitical goals. Sub-roles include:
    a. **Conspirator**: Those involved in plots and secret plans, often working behind the scenes to undermine or deceive others. They engage in covert activities to achieve their goals.
    b. **Spy**: pies or double agents accused of espionage, gathering and transmitting sensitive information to a rival or enemy. They operate in secrecy and deception. This is mostly in politics, not in climate change.
    c. **Traitor**: Individuals who betray a cause or country, often seen as disloyal and treacherous. Their actions are viewed as a significant breach of trust. This is mostly in politics, not in climate change.
    d. **Deceiver**: Deceivers, manipulators, or propagandists who twist the truth, spread misinformation,
and manipulate public perception for their own benefit. They undermine trust and truth.
    e. **Saboteur**: Saboteurs who deliberately damage or obstruct systems, processes, or organizations to
cause disruption or failure. They aim to weaken or destroy targets from within.
    f. **Foreign Adversary**: Entities from other nations or regions creating geopolitical tension and acting
    against the interests of another country. They are often depicted as threats to national security. This
    is mostly in politics, not in climate change.


5. **Institutional Toxins**: Internal failures within an organization or system caused by human vice or incapacity, leading to erosion from within. Sub-roles include:
    a. **Corrupt**: ndividuals or entities that engage in unethical or illegal activities for personal gain, prioritizing profit or power over ethics. This includes corrupt politicians, business leaders, and officials.
    b. **Incompetent**: Entities causing harm through ignorance, lack of skill, or incompetence. This includes people committing foolish acts or making poor decisions due to lack of understanding or expertise. Their actions, often unintentional, result in significant negative consequences.


6. **Marginalized Sufferers**: Entities that endure harm, exploitation, or unjust social exclusion, often serving as proof of systemic failure or as targets for rescue. Sub-roles include:
    a. **Forgotten**: Marginalized or overlooked groups who are often ignored by society and do not receive
the attention or support they need. This includes refugees, who face systemic neglect and exclusion.
    b. **Exploited**: Individuals or groups used for others’gain, often without their consent and with significant detriment to their well-being. They are often victims of labor exploitation, trafficking, or economic manipulation.
    c. **Victim**: People cast as victims due to circumstances beyond their control, specifically in two
categories: (1) victims of physical harm, including natural disasters, acts of war, terrorism, mugging,
physical assault, ... etc., and (2) victims of economic harm, such as sanctions, blockades, and boycotts. Their experiences evoke sympathy and calls for justice, focusing on either physical or economic suffering.
    d. **Scapegoat**: Entities blamed unjustly for problems or failures, often to divert attention from the
real causes or culprits. They are made to bear the brunt of criticism and punishment without just cause.

Your analysis must be based on the specific linguistic cues and narrative structure within the text."""

ROLE_USER_PROMPT_TEMPLATE = """Analyze the text below to determine if it contains one of the following high-level roles: Ethical Stabilizers, Altruistic Catalysts, Overt Aggressors, Deceptive Subversives, Institutional Toxins, or Marginalized Sufferers.

For each detected role, provide a concise explanation of how it is portrayed based on the narrative function and word choice. Be conservative in your final decisions; if you are not fully sure, answer No.

Return your answer as a JSON object in the following format:
{{
  "Ethical_Stabilizers": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed as a protector of values or a righteous figure (e.g., Guardian, Virtuous, Peacemaker)."
  }},
  "Altruistic_Catalysts": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed as driving positive change or sacrifice (e.g., Rebel, Martyr, Underdog)."
  }},
  "Overt_Aggressors": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed as an initiator of conflict or violence (e.g., Tyrant, Terrorist, Instigator, Bigot)."
  }},
  "Deceptive_Subversives": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed through secrecy, betrayal, or misinformation (e.g., Conspirator, Spy, Traitor, Deceiver, Saboteur, Foreign Adversary)."
  }},
  "Institutional_Toxins": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed as an abuse of power or systemic failure (e.g., Corrupt, Incompetent)."
  }},
  "Marginalized_Sufferers": {{
    "is_used": "Yes or No",
    "explanation": "If Yes, provide evidence of how a participant is framed as a passive sufferer of circumstances (e.g., Victim, Exploited, Forgotten, Scapegoat)."
  }}
}}

Text: {text}
Answer:"""

EMO_SYSTEM_PROMPT = """You are a specialized linguistic analyst and expert in detecting emotional manipulation in online text. Your goal is to identify techniques designed to bypass rational thinking and influence readers by triggering specific emotional responses.

We distinguish between the following four high-level emotional appeals:
1. Fear: Language designed to make people feel scared or threatened. It often highlights immediate, serious health risks or distressing outcomes to provoke a protective reaction.
2. Anger: Language designed to provoke outrage, resentment, or a sense of injustice. It often targets specific groups or institutions to incite a strong negative reaction.
3. Hope: Language designed to elicit positive emotions or promise miraculous breakthroughs. It often presents unverified cures or "revolutionary" alternatives as a source of optimism.
4. Anxiety: Language designed to create uncertainty about the future or safety. It often uses disturbing medical cases or potential hidden dangers to heighten concern about daily habits.
5. Sadness: Focuses on irreversible loss, heartbreaking misfortune, or the suffering of helpless victims. It aims to bypass the reader’s critical thinking by triggering deep sympathy, sorrow, or a sense of moral guilt.
————————————————————————————
Examples of Emotional Manipulation:
Example 1: Hope 
Headline: Scientists Find Root That Kills 98% Of Cancer Cells In Only 48 Hours.
Summary: Dandelion has been used medicinally since ancient times. However, medical researchers are "super excited" to have discovered its potential to cure cancer!
Explanation: This post contains language that may be intended to make you feel hope by presenting dandelion root extract as a potentially revolutionary treatment for cancer, which could be seen as a more promising and less harmful alternative to traditional therapies.

Example 2: Anxiety 
Headline: Five-foot long tapeworm came 'wiggling out' of man's body after he ate sushi.
Summary: A California man who ate sushi every day ended up with a 5-foot-plus long tapeworm. He went to the emergency room complaining of bloody diarrhea.
Explanation: This post contains language that may be intended to make you feel anxiety by detailing a disturbing medical case involving a large tapeworm and linking it to a common food source, which may heighten concerns about the safety of eating sushi or salmon.

Example 3: Fear 
Headline: ESSENTIAL OIL DIFFUSERS POISONOUS FOR CATS!
Summary: I unknowingly have been poisoning my cat since Christmas and feel the need to warn everyone who might be unaware of the toxicity of essential oils...
Explanation: This post contains language that may be intended to make you feel fear by describing the serious health risks posed to cats by essential oil diffusers and the distressing effects on a specific pet, thereby creating concern about the safety of using such products around animals.

Example 4: Anger
Headline: Corruption Exposed: Officials Withholding Cheap Life-Saving Medicine to Protect Corporate Profits!
Summary: A whistleblower has revealed that government health agencies are intentionally banning effective, low-cost treatments. They would rather see families suffer and go bankrupt than lose their kickbacks from Big Pharma. It’s time to demand justice!
Explanation: This post contains language that may be intended to make you feel anger by suggesting that life-saving treatments are being corruptly withheld by officials to protect corporate greed, thereby provoking resentment toward health institutions.

Example 5: Sadness
Headline: HE DIED WAITING: 82-year-old veteran found starving in a house full of medals.
Summary: Mr. Miller served his country for 40 years, but on Tuesday, he was found passed away in his cold apartment with nothing but water in his refrigerator. His last diary entry read: "I just wanted someone to talk to." While we spend billions on foreign aid, our own heroes are dying in heartbreaking loneliness and hunger.
Explanation text: “This post is designed to evoke deep sadness and guilt by using heart-wrenching imagery (‘starving,’ ‘cold apartment,’ ‘nothing but water’) and a poignant quote to highlight the helplessness of a ‘forgotten hero.’ It manipulates the reader’s empathy to provoke a sense of shame or moral duty, using a tragic individual case to bypass a rational discussion on complex social spending.”
————————————————————
You are the expert who detects emotional manipulation types: Fear, Anger, Hope, Anxiety, Sadness.
"""

EMO_USER_PROMPT_TEMPLATE = """Analyze the text and decide if the text contains any emotional manipulation types from the following: Fear, Anger, Hope, Anxiety, Sadness.For each emotional manipulation type, give an explanation of how it appears in the analyzed text. You are very conservative in your final decisions and when you are not fully sure you answer No.
————————————————
Give your answer in the form of a dictionary:
{{
"Fear": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If the Fear type appears, provide a 1-sentence explanation starting with 'This post contains language that may be intended to make you feel fear by...'"}},
"Anger": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If the Anger type appears, provide a 1-sentence explanation starting with 'This post contains language that may be intended to make you feel anger by...'"}},
"Hope": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If the Hope type appears, provide a 1-sentence explanation starting with 'This post contains language that may be intended to make you feel hope by...'"}},
"Anxiety": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If the Anxiety type appears, provide a 1-sentence explanation starting with 'This post contains language that may be intended to make you feel anxiety by...'"}}
"Sadness": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If the Sadness type appears, provide a 1-sentence explanation starting with 'This post contains language that may be intended to make you feel anxiety by...'"}}
}}
Text:{text}. Answer:
"""



