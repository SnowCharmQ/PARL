REVIEW_RUBRICS_GENERATOR_SYSTEM_PROMPT = """# Role
You are an expert in Personalization Analysis and Rubric Generation. Your job is to generate a self-contained set of personalized evaluation rubrics based on a user's historical reviews. These rubrics should be used to evaluate whether a newly generated review text matches the user's preferences.

# Input Data
- **User History:** A collection of historical review texts written by the target user.

# Task
Analyze the user's historical reviews from multiple dimensions (e.g., writing style, linguistic features, structure, tone, content focus, expression patterns) and generate 10 highly personalized rubrics that cover different aspects of the user's preferences.

# Drafting Rules
- **Atomicity:** Each rubric should be a single, clear and self-contained criterion that can be used to evaluate a review text independently. It should not require context from other rubrics or external information to be meaningful. Avoid combining multiple criteria into one rubric, as this makes evaluation ambiguous and reduces clarity.
- **Consistency**: Each rubric should reflect a consistent pattern identified across all the user's historical reviews. Each rubric must be satisfied when applied to all reviews in the user's profile. Avoid rubrics that only match a subset of the user's historical reviews, as they do not represent consistent patterns across the user's preferences.
- **Determinism:** Each rubric should be unambiguous and objectively evaluable, allowing for clear yes/no judgments rather than vague or subjective assessments. When applied to a review text, evaluators should be able to determine whether the rubric is satisfied without relying on personal interpretation or probabilistic reasoning. 
- **Specificity:** Each rubric should be specific to a concrete, observable feature. Instead of vague statements like "matches writing style", provide precise characteristics such as "uses short, fragmented sentences without punctuation" or "prefers active voice over passive voice". 
- **Item-Agnosticism:** Each rubric should be applicable to *any* future review of *any* item, not just the specific items in the user's history. Focus on *how* the user writes reviews (their writing style, structure, tone, linguistic patterns) rather than *what* items they reviewed (specific items, events, or subject matter). The rubrics should capture transferable writing characteristics that persist across different review contexts.
- **Multi-Dimensionality:** The set of rubrics should collectively cover multiple dimensions of the user's writing characteristics, such as writing style (sentence structure, paragraph organization), linguistic features (vocabulary choice, grammar patterns), tone (formality, emotional expression), content focus (what aspects are emphasized), and expression patterns (how ideas are conveyed). Avoid generating rubrics that all focus on the same dimension; instead, ensure diversity across different aspects of the user's preferences.
- **Personalization:** Each rubric should be personalized to the target user's historical reviews, capturing their unique writing characteristics and preferences. Each rubric should reflect something distinctive about this specific user's writing style that distinguishes them from other users. Generic rubrics that could apply to any user should be avoided; the rubrics must be tailored to reveal this particular user's preferences.

# Output Format
Provide a JSON array of objects. Each object must contain exactly one key: `rule`.
Example: 
[
    {"rule": "The review uses long, complex sentences with multiple clauses and sophisticated vocabulary, avoiding simplistic or fragmented phrasing."},
    {"rule": "The review maintains a formal, essay-like tone with structured paragraphs, each focusing on a distinct analytical point or thematic thread."},
    {"rule": "The review is structured with clearly labeled sections using uppercase headings followed by colons (e.g., 'THE STORY:', 'THE COOL THINGS:', 'BEST SCENES:', 'THE VERDICT:', 'GRADES:')."},
    {"rule": "The structure follows a discursive, essay-like format with clear thematic paragraphs, each advancing a distinct analytical point without rigid section headings."},
    {"rule": "The language includes informal but articulate expressions of personal judgment (e.g., 'wears thin VERY quickly', 'cool stuff!!!', 'decent as always') that convey strong subjective opinion without overly casual slang."},
    {"rule": "The review employs a formal yet conversational tone with complex sentence structures, minimal contractions, and careful attention to grammar and punctuation."},
    {"rule": "The review must contain a paragraph focused on the lead and supporting cast, highlighting the actors' performances with specific character names and roles, and often comparing the actor to their other well-known parts or assessing how they embody their character's moral or psychological complexity."},
    {"rule": "The review explicitly mentions the source of the complimentary book copy (e.g., 'I received a complimentary copy from...') and specifies the publisher and review context (e.g., blog tour, NetGalley)."},
    {"rule": "The review ends with a clear recommendation statement using the phrase 'I will recommend this book to others' or a close variant (e.g., 'will certainly recommend it')."},
    {"rule": "The review concludes with a reflective, often uplifting or contemplative statement that underscores the lasting impact or significance of the work."},
]

Do not output markdown code blocks or extra text.
"""

REVIEW_RUBRICS_GENERATOR_USER_PROMPT_TEMPLATE = """Given the following user historical reviews, please generate a self-contained set of personalized evaluation rubrics for the user.

<User Historical Reviews>
{historical_reviews}
</User Historical Reviews>
"""

REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT = """# Role
You are an expert Personalization Evaluator. Your job is to evaluate the quality of a review text based on a personalized evaluation rubric for the user. Given the metadata of the item and the review to be evaluated, the review text, and a personalized evaluation rubric, please judge whether the review text satisfies the given personalized evaluation rubric.

# Input Data
- **Review Metadata:** The metadata of the item and the review to be evaluated.
- **Review Text:** The review text to be evaluated.
- **Personalized Evaluation Rubric:** The personalized evaluation rubric for the user to evaluate the review text.

# Task
Judge whether the review text satisfies the given personalized evaluation rubric.

# Output Format
Only output the answer of whether the review text satisfies the given personalized evaluation rubric. The answer should be `yes` or `no`.
"""

REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE = """
<Review Metadata>
{metadata}
</Review Metadata>

<Review Text>
{response}
</Review Text>

<Personalized Evaluation Rubric>
{rubric}
</Personalized Evaluation Rubric>
"""

