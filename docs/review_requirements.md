# FoodSense — Post-Review Requirements

Mandate for the repositioning of FoodSense, issued following the project review.

SOURCE_NOTE: Text as supplied by the project guide, Dr. Thanuja R, following the
review. Sections 1-50 are reproduced without any alteration to wording. The text
reached this repository via the architect working session rather than from an
original file on disk; in transfer, line breaks in sections 37-50 were collapsed
and have been restored. Whitespace and section separators in that range are
therefore reconstructed, not original. The agent-directed preamble that preceded
section 1 in the delivered text has been removed as a delivery wrapper, not a
requirement.

RECEIVED: <YYYY-MM-DD>

VERIFIED_BY: <name, date — sections 37-50 checked against the original as sent>

Execution is governed by `docs/pivot.md`. Where this document and the architect's
rulings differ, this document governs what must exist and the rulings govern how.

---

1. FINAL PROJECT OBJECTIVE

The final application should be:

An Evidence-Grounded RAG-Based Personalized Food Recommendation System for Toddlers with Nutrition-Related Health Conditions Using Indian Regional Food Data.

The system should combine:

- Toddler-specific nutritional requirements
- Disease/condition-specific knowledge
- Indian food/nutrition data
- North Indian cuisine
- South Indian cuisine
- RAG
- Vector search
- LLM-based explanation
- Nutritional analysis
- Dynamic food database
- Evidence/source tracking
- Research-quality visualizations
- Modern responsive UI

The system should provide personalized food recommendations based on:

Toddler + Condition + Nutritional Requirements + Indian Cuisine + Food Data + Retrieved Evidence

---

2. VERY IMPORTANT — TODDLERS ONLY

The application must focus EXCLUSIVELY on:

TODDLERS / YOUNG CHILDREN

Do NOT build the recommendation system for:

- Older adults
- Adults
- Adolescents
- General population

Remove older-adult recommendation logic if it currently exists, unless it is required for backward compatibility.

The UI and research positioning should clearly communicate that the target population is toddlers.

Use the project's existing definition of "toddler" if already established.

If an age range needs to be defined, use an authoritative pediatric/nutrition guideline rather than arbitrarily selecting an age range.

---

3. DISEASE / HEALTH CONDITION SCOPE

The system should be condition-specific, but conditions must be relevant to toddlers/young children.

Do NOT make diabetes the focus.

Diabetes should NOT be the primary disease category.

Do NOT simply copy common adult nutrition diseases into the toddler system.

Potential toddler-relevant conditions include:

- Iron-deficiency anemia
- Protein-energy malnutrition / undernutrition
- Vitamin D deficiency
- Calcium deficiency / nutritional bone-health concerns
- Constipation
- Celiac disease
- Food allergy-related dietary restrictions
- Poor weight gain / growth-related nutritional concerns where supported by reliable evidence

However, do NOT blindly implement every condition listed above.

The final supported conditions must be based on credible pediatric evidence and authoritative sources.

The architecture must allow additional toddler-specific conditions to be added later without rewriting the application.

---

4. DIABETES EXCLUSION

Do NOT position the project as a diabetes food recommendation system.

Do NOT make diabetes the main demonstration disease.

If diabetes functionality already exists in the current project, preserve it only if removing it would break the application, but the redesigned research scope should clearly prioritize toddler-specific nutrition-related conditions other than diabetes.

The primary research experiments and UI should use toddler-relevant conditions.

---

5. EXISTING PROJECT — INSPECT FIRST

Before modifying anything, inspect:

- Project directory structure
- Frontend framework
- Backend framework
- Database
- Existing USDA integration
- Existing food schema/model
- Existing RAG implementation
- Embedding model
- Vector database/vector store
- Retrieval pipeline
- LLM/API integration
- Recommendation logic
- Existing disease logic
- Nutritional calculations
- Existing charts
- Existing UI components
- Authentication
- API routes
- Environment configuration
- Deployment configuration

Understand the existing data flow before making architectural changes.

Reuse existing technologies wherever practical.

Do not introduce a new framework just because it is available.

---

6. CURRENT USDA DATASET PROBLEM

The existing system currently relies heavily on the USDA dataset.

USDA is useful as a supplementary nutritional source, but it does not sufficiently represent Indian regional cuisine.

Therefore:

DO NOT DELETE USDA

Instead, create a unified multi-source food knowledge architecture.

The application should support:

USDA Food Data
+
Indian Food Dataset
+
User-Added Food
+
Future Verified Sources

Every food should retain its source/provenance.

---

7. INDIAN FOOD DATASET — HIGH PRIORITY

Add a properly sourced Indian food dataset.

The dataset should include foods from both:

NORTH INDIAN CUISINE

Examples:

- Roti
- Chapati
- Paratha
- Aloo Paratha
- Naan
- Dal
- Dal Tadka
- Dal Makhani
- Rajma
- Chole
- Kadhi
- Paneer
- Paneer Tikka
- Khichdi
- Biryani
- Samosa
- Pakora
- Curd
- Lassi
- Appropriate fruit/vegetable preparations
- Other commonly consumed regional foods

SOUTH INDIAN CUISINE

Examples:

- Idli
- Plain Dosa
- Masala Dosa
- Ragi Dosa
- Ragi Idli
- Medu Vada
- Sambar
- Rasam
- Pongal
- Upma
- Uttapam
- Appam
- Puttu
- Curd Rice
- Lemon Rice
- Tamarind Rice
- Coconut Rice
- Appropriate vegetable preparations
- Fruit-based foods
- Other commonly consumed regional foods

These are examples, NOT a complete hard-coded list.

The database must be designed to scale.

---

8. TODDLER-APPROPRIATE FOOD DATA

Do not simply take an adult Indian food dataset and assume every item is appropriate for toddlers.

Where possible, store metadata related to toddler suitability.

Potential fields:

- Age appropriateness
- Texture/consistency
- Preparation method
- Choking-risk notes where reliable
- Allergen information where available
- Added sugar information
- Sodium information
- Ingredients
- Vegetarian/non-vegetarian status where reliable

Do not make unsupported medical or food-safety claims.

If sufficient evidence is unavailable, mark the information as unknown rather than inventing it.

---

9. UNIFIED FOOD DATA MODEL

Create/improve a structured food model.

A food record should ideally support:

- id
- name
- aliases
- cuisine
- region
- category
- vegetarian/non-vegetarian if reliable
- toddler suitability metadata where available
- serving_size
- serving_unit
- calories
- protein
- carbohydrates
- fat
- fiber
- sugar
- sodium
- potassium
- iron
- calcium
- vitamin D
- other micronutrients where available
- ingredients
- preparation_method
- source
- source_reference
- source_type
- confidence
- verification_status
- created_at
- updated_at

Only store nutrients that are actually available from reliable data.

DO NOT fabricate nutritional values.

---

10. DATA PROVENANCE

Because this is a research-oriented system, provenance is mandatory.

Each food record should clearly indicate its origin.

Possible source types:

- USDA
- Indian Dataset
- Research Dataset
- Government Source
- Verified Source
- User Added

For user-added foods, clearly label them as user-provided.

Do not present user-entered nutritional information as authoritative medical/nutritional evidence.

Allow users to provide, where appropriate:

- Source name
- Source URL/reference
- Dataset name
- Notes

Never fabricate citations or references.

---

11. TODDLER-SPECIFIC DISEASE KNOWLEDGE BASE

Create a structured disease/condition knowledge layer.

Each condition should contain information such as:

- Condition name
- Description
- Target population
- Relevant nutrients
- Nutrients to monitor
- Dietary considerations
- Foods/nutrients that may be relevant
- Restrictions where evidence supports them
- Contraindications where applicable
- Pediatric considerations
- Evidence/source
- Source reference

The system should prioritize toddler/child-specific evidence.

Do NOT take adult nutritional guidelines and directly apply them to toddlers.

---

12. RAG ARCHITECTURE

The project must genuinely use Retrieval-Augmented Generation.

The system should NOT simply be:

User question → LLM → answer

Instead use:

User Input
↓
Identify toddler age/context
↓
Identify health condition
↓
Extract dietary requirements
↓
Retrieve toddler-specific disease evidence
↓
Retrieve Indian food/nutritional data
↓
Retrieve relevant food safety/allergen information where available
↓
Rank/filter candidate foods
↓
Generate recommendation using retrieved context
↓
Validate recommendation
↓
Return recommendation + reasoning + sources + nutritional information
↓
Generate visualization where applicable

---

13. RAG KNOWLEDGE SOURCES

The RAG knowledge base should prioritize credible sources such as:

- Pediatric nutrition guidelines
- Government/public-health resources
- Peer-reviewed research
- Pediatric organizations
- Nutrition science literature
- Reliable Indian nutritional sources
- Properly sourced Indian food datasets

Each document should ideally maintain:

- Source
- Title
- URL/reference
- Publication information where available
- Population
- Condition
- Relevant nutrients
- Document type

Do NOT fabricate research papers, sources, URLs, or citations.

---

14. RETRIEVAL MUST BE TODDLER-SPECIFIC

This is critical.

Do not retrieve generic:

"foods for anemia"

when the actual query is:

"foods for iron-deficiency anemia in toddlers."

Retrieval should prioritize:

Condition + Toddler + Nutritional Requirement

For example:

User:
"Suggest Indian foods for a toddler with iron-deficiency anemia."

RAG should retrieve:

1. Toddler-specific nutrition requirements
2. Pediatric iron-deficiency evidence
3. Indian food nutritional composition
4. Relevant North/South Indian foods
5. Relevant food safety information

Then generate the answer.

---

15. AGE + CONDITION INTERSECTION

The recommendation must be based on the intersection of:

AGE + CONDITION + NUTRITION + FOOD + EVIDENCE

Do not treat age as a cosmetic filter.

Age should actually influence:

- Retrieval
- Nutritional interpretation
- Candidate ranking
- Recommendation generation
- Safety considerations

---

16. RECOMMENDATION ENGINE

The recommendation engine should consider:

1. Toddler age
2. Selected health condition
3. Nutritional requirements
4. Food nutritional profile
5. Indian cuisine preference
6. Dietary restrictions
7. Allergens if known
8. Evidence retrieved by RAG
9. Serving size
10. Food safety considerations

Recommendations should be explainable.

Example:

Recommended Food

Ragi Idli

Why?

Explain why the food was selected based on actual nutritional data and retrieved evidence.

Nutritional Profile

Calories
Protein
Fiber
Iron
Calcium
etc.

Cuisine

South Indian

Serving Basis

Clearly state serving size.

Evidence

Show relevant retrieved sources.

Do not make claims stronger than the evidence supports.

---

17. MEDICAL SAFETY

This is a nutritional decision-support system, NOT a diagnostic or treatment system.

The system must never claim:

- To diagnose a disease
- To cure a disease
- To replace a pediatrician
- To replace a registered dietitian
- That a food will medically treat a condition

If evidence is insufficient, say so.

For high-risk conditions, allergies, medication interactions, severe malnutrition, significant growth concerns, or other clinically sensitive situations, clearly recommend consultation with a qualified healthcare professional.

---

18. TODDLER FOOD SAFETY

Where reliable evidence is available, consider:

- Choking hazards
- Texture
- Preparation method
- Portion size
- Potential allergens
- Excessive salt
- Added sugar
- Food preparation safety

Do not invent safety classifications.

If the system does not have sufficient information, display:

"Insufficient evidence available for this safety attribute."

---

19. DYNAMIC FOOD ADDITION

This is one of the MOST IMPORTANT features.

If a user enters a food that does not exist:

Example:

"Ragi Dosa"

System:

Search database
↓
Food not found
↓
Show "Add Food"
↓
User enters food information
↓
Validate
↓
Check duplicate/similar foods
↓
Save permanently
↓
Update food list
↓
Generate embedding
↓
Update vector database
↓
Food becomes searchable
↓
Food becomes available to RAG
↓
Future users can retrieve the food

This must persist after page refresh/restart.

Do NOT store new food only in frontend state/local temporary memory unless the existing architecture explicitly requires it.

Use the project's persistent database/backend.

---

20. USER-ADDED FOOD → RAG PIPELINE

Whenever a new food is added:

1. Store structured nutritional data.
2. Store metadata.
3. Normalize the food information.
4. Generate a text representation.
5. Generate its embedding.
6. Add it to the vector store.
7. Make it available for retrieval.

Example:

Food:
Ragi Dosa

Cuisine:
South Indian

Region:
South India

Nutrients:
...

Source:
User Added / provided reference

This should become a retrievable RAG document.

---

21. FOOD UPDATE

If a user/admin updates a food:

Database
↓
Update record
↓
Regenerate normalized representation
↓
Regenerate embedding
↓
Replace old vector
↓
Update vector store

Database and vector store must remain synchronized.

---

22. DUPLICATE DETECTION

Prevent duplicate entries.

For example:

Masala Dosa
masala dosa
Masala-Dosa

should not automatically become three independent foods.

Implement:

- Case normalization
- Whitespace normalization
- Alias matching
- Fuzzy matching
- Unique IDs

Before adding:

"Similar food already exists. Did you mean Masala Dosa?"

---

23. DATA VALIDATION

Validate all user-added nutritional information.

Examples:

- Calories >= 0
- Protein >= 0
- Carbohydrates >= 0
- Fat >= 0
- Fiber >= 0
- Sodium >= 0
- Serving size > 0
- Numeric fields must contain valid numbers
- Required fields must be present

Do not allow invalid data to enter the main knowledge base.

---

24. FOOD SEARCH

Improve food search.

Users should be able to search:

- Food name
- Alias
- Cuisine
- Region
- Category

Examples:

"Idli"
"South Indian breakfast"
"Ragi"
"North Indian"
"high protein"

depending on available functionality.

Use appropriate search normalization and fuzzy matching.

---

25. FOOD FILTERING

Add useful filters such as:

- Cuisine
- Region
- Food category
- Vegetarian/non-vegetarian if reliable
- Nutritional ranges
- Calories
- Protein
- Fiber
- Iron
- Calcium
- Other available nutrients

Keep the interface simple.

---

26. FOOD COMPARISON

Add/improve a food comparison feature.

Example:

Select:

- Idli
- Masala Dosa
- Ragi Idli
- Aloo Paratha

Then compare:

- Calories
- Protein
- Carbohydrates
- Fat
- Fiber
- Iron
- Calcium
- Sodium
- Other available relevant nutrients

Allow users to select the comparison metric.

Example:

Compare by:

- Calories
- Protein
- Fiber
- Iron
- Calcium

---

27. NORMALIZATION

Do NOT compare incompatible serving bases.

For research-quality comparison, prefer:

Per 100g

where the underlying data allows it.

Otherwise use:

Per serving

and clearly display the serving size.

Every chart/table must clearly indicate its normalization basis.

---

28. RESEARCH-QUALITY GRAPH / VISUALIZATION

Add a professional visualization section suitable for a research paper.

Do NOT create a flashy generic dashboard chart.

The graph should be clean, scientific, publication-friendly, and easy to interpret.

Support appropriate visualizations such as:

- Grouped bar chart
- Nutrient comparison chart
- Multi-food comparison
- Nutrient profile chart
- Radar chart where scientifically appropriate

Possible metrics:

- Calories
- Protein
- Carbohydrates
- Fat
- Fiber
- Iron
- Calcium
- Sodium
- Potassium
- Vitamin D
- Other available nutrients

The chart must have:

- Clear title
- Proper axis labels
- Correct units
- Legend
- Food names
- Consistent typography
- Appropriate scaling
- Clean background
- Minimal visual clutter
- Responsive layout
- Research-friendly presentation

If practical, provide export functionality:

- PNG
- SVG
- PDF

Do not fabricate or transform values in a misleading way.

---

29. RESEARCH VISUALIZATION EXAMPLE

For example:

Nutritional Comparison of Selected Indian Foods

X-axis:
Food

Y-axis:
Nutrient value

Groups:
Protein / Fiber / Iron / Calcium

Clearly mention:

"Values normalized per 100g"

or:

"Values calculated per serving"

Also include the data source.

The visualization should be good enough to use as a figure in a research paper after appropriate academic formatting.

---

30. RESEARCH ANALYSIS

Where technically appropriate, provide analysis such as:

- Highest protein food
- Highest fiber food
- Highest iron food
- Highest calcium food
- Lowest/highest calorie food
- Nutrient density comparison

But only calculate metrics from actual stored data.

Do not generate conclusions that are not supported by the dataset.

---

31. RAG EVALUATION

Because this is a research project, implement or prepare an evaluation framework.

Possible retrieval metrics:

- Precision@K
- Recall@K
- MRR

Possible RAG metrics:

- Context relevance
- Answer relevance
- Faithfulness / groundedness

Possible recommendation metrics:

- Nutritional constraint satisfaction
- Recommendation relevance
- Evidence-grounded recommendation rate

Do NOT invent scores.

If a labeled evaluation dataset does not currently exist:

- Build the evaluation framework.
- Provide sample evaluation structure if appropriate.
- Clearly state that actual numerical evaluation requires a proper labeled test set.

---

32. RAG TRANSPARENCY / EXPLAINABILITY

The system should show why a food was recommended.

For each recommendation, provide:

- Food
- Relevant nutrients
- Retrieved evidence
- Reasoning
- Source/reference

The user/researcher should be able to answer:

"Why did the system recommend this food?"

Do not expose hidden chain-of-thought.

Instead provide concise, evidence-based reasoning.

---

33. MODERN UI/UX

Redesign the current UI without breaking functionality.

The application should look like a polished:

Pediatric Nutrition + AI Research Analytics Platform

Improve:

- Typography
- Layout
- Navigation
- Cards
- Forms
- Buttons
- Search
- Tables
- Charts
- Recommendation cards
- Source/evidence display
- Empty states
- Loading states
- Error states
- Success states

Use a consistent design system.

Avoid unnecessary visual clutter.

Animations should be subtle and purposeful.

---

34. RECOMMENDATION USER FLOW

Create a simple flow:

Step 1 — Toddler Information

Select/enter:

- Age
- Other required nutritional context supported by the project

Step 2 — Condition

Select a supported toddler-specific condition.

Examples:

- Iron-deficiency anemia
- Undernutrition
- Vitamin D deficiency
- Constipation
- Celiac disease
- etc.

Step 3 — Cuisine

Choose:

- North Indian
- South Indian
- Both

Step 4 — Dietary Preferences/Restrictions

Where applicable:

- Vegetarian
- Non-vegetarian
- Allergies
- Other supported restrictions

Step 5 — Generate Recommendation

Display:

- Recommended foods
- Nutritional values
- Explanation
- Evidence
- Sources
- Comparison chart

---

35. MOBILE RESPONSIVENESS

The entire application must work properly on:

- Android phones
- iPhones
- Tablets
- Laptops
- Desktops

Test approximately:

320px
375px
390px
414px
768px
1024px
1440px

Ensure:

- No broken layouts
- No accidental horizontal overflow
- Responsive cards
- Responsive forms
- Responsive charts
- Readable tables
- Mobile-friendly navigation
- Proper touch targets

---

36. ERROR HANDLING

Handle gracefully:

- Food not found
- Condition not found
- Missing nutritional information
- Missing RAG evidence
- Vector database failure
- LLM/API failure
- Database failure
- Invalid food data
- Duplicate food
- Unsupported condition
- Insufficient evidence

Never show false success.

Never expose raw backend stack traces to normal users.

If RAG retrieval is insufficient:

Do not hallucinate.

Return an appropriate message explaining that sufficient evidence was not found.

---

37. PERFORMANCE

Optimize:

Search
Database queries
Vector retrieval
Embedding generation
LLM calls
Chart rendering

Use where appropriate:

Debounced search
Pagination
Lazy loading
Efficient queries
Caching
Batch operations

Do not over-engineer.

---

38. SECURITY

If authentication exists:

Protect food-management operations.
Protect API keys.
Validate backend inputs.
Sanitize user input.
Prevent unauthorized database modification.

If the application allows public food additions, consider:

Pending
Verified
Rejected

verification states.

This is particularly valuable for a research-oriented database.

---

39. IMPORTANT DATA INTEGRITY RULES

NEVER fabricate:

Nutritional values
Disease information
Medical recommendations
Research papers
Sources
Citations
Dataset statistics
Evaluation scores

Clearly distinguish:

Verified Data vs. User-Added Data vs. Retrieved Evidence vs. LLM-Generated Explanation

---

40. END-TO-END EXAMPLE

The final system should support a query like:

"Suggest South Indian foods for a toddler with iron-deficiency anemia."

The system should:

Identify target population = Toddler.
Identify condition = Iron-deficiency anemia.
Identify cuisine = South Indian.
Retrieve toddler-specific anemia evidence.
Retrieve Indian nutritional data.
Retrieve suitable South Indian foods.
Apply nutritional/data filters.
Generate evidence-grounded recommendations.
Display nutritional values.
Explain why each food was recommended.
Show evidence/source references.
Display a comparison chart.

The answer must NOT be a generic LLM response.

---

41. ANOTHER EXAMPLE

User:

"My toddler has constipation. Show suitable North Indian food options."

System:

Toddler + Constipation + North Indian

→ Retrieve pediatric evidence
→ Retrieve Indian food data
→ Analyze fiber/nutrition where relevant
→ Recommend suitable foods
→ Explain reasoning
→ Display sources
→ Visualize nutritional comparison

Again, no unsupported medical claims.

---

42. NEW FOOD EXAMPLE

User enters:

"Ragi Dosa"

System searches.

If it exists:

→ Display existing record.

If it does not:

→ "Food not found"
→ "Add Food"

User enters:

Name: Ragi Dosa
Cuisine: South Indian
Region: South India
Serving: 100g
Nutrition: [actual values]
Source: [actual source]

Then:

Validate
↓
Duplicate check
↓
Save to database
↓
Generate embedding
↓
Update vector store
↓
Refresh food list
↓
Make available to RAG

After another user searches for "Ragi Dosa", it should already be present.

No source-code modification should be required.

---

43. ARCHITECTURAL PRINCIPLE

Keep structured and unstructured knowledge separate but connected.

For example:

STRUCTURED DATABASE
→ Nutritional values
→ Food metadata
→ Cuisine
→ Region
→ Serving size
→ Source

VECTOR DATABASE
→ Food descriptions
→ Disease knowledge
→ Research documents
→ Nutritional context
→ User-added food representations

RAG SYSTEM
→ Retrieves relevant evidence from both

RECOMMENDATION ENGINE
→ Applies nutritional/contextual filtering

LLM
→ Generates understandable evidence-grounded explanation

VISUALIZATION
→ Presents quantitative results

---

44. DO NOT TURN IT INTO A GENERIC CHATBOT

The main research contribution should be the complete pipeline:

User Context
↓
Toddler-Specific Knowledge
↓
Disease/Condition Retrieval
↓
Indian Food Retrieval
↓
Nutritional Filtering
↓
Recommendation
↓
Evidence-Grounded Generation
↓
Explanation
↓
Visualization

The LLM should not be the only intelligence in the system.

---

45. FINAL RESEARCH POSITIONING

The project should be positioned around:

Toddler-specific personalized nutrition recommendations for health-related dietary conditions using RAG and Indian regional food data.

The key technical contributions should include:

RAG
NLP/LLM
Vector search
Pediatric/toddler-specific knowledge retrieval
Indian food knowledge
North/South Indian cuisine
Disease/condition-specific recommendation
Nutritional analysis
Dynamic knowledge-base updates
Explainability
Data provenance
Research-quality visualization
Evaluation framework

---

46. IMPLEMENTATION ORDER

Follow this order:

Phase 1
Inspect existing project.

Phase 2
Improve food database architecture.

Phase 3
Integrate properly sourced Indian food data.

Phase 4
Create toddler-specific condition knowledge base.

Phase 5
Improve RAG retrieval pipeline.

Phase 6
Implement toddler + condition personalization.

Phase 7
Implement dynamic food addition.

Phase 8
Automatically embed/index new foods.

Phase 9
Implement food comparison.

Phase 10
Implement research-quality charts.

Phase 11
Improve UI/UX.

Phase 12
Optimize mobile responsiveness.

Phase 13
Implement validation/error handling/security.

Phase 14
Implement RAG/recommendation evaluation framework.

Phase 15
Run complete end-to-end testing.

---

47. TESTING REQUIREMENTS

Test at minimum:

Test 1
Toddler + Iron-deficiency anemia + North Indian

Test 2
Toddler + Iron-deficiency anemia + South Indian

Test 3
Toddler + Constipation + North Indian

Test 4
Toddler + Vitamin D deficiency + South Indian

Test 5
Toddler + Undernutrition

Test 6
Toddler + Celiac disease

Test 7
Unknown food → Add food → Refresh → Food persists

Test 8
New food → Embedding generated → RAG retrieves it

Test 9
Two foods → Comparison graph

Test 10
Multiple foods → Research-quality graph

Test 11
Insufficient evidence → No hallucinated recommendation

Test 12
Duplicate food → Duplicate warning

Test 13
Invalid nutrition values → Validation error

Test 14
Mobile screen → UI remains usable

---

48. FINAL UI REQUIREMENT

The final interface should communicate immediately that this is:

AI + RAG + Toddler Nutrition + Indian Food + Evidence-Based Recommendation

It should look suitable for:

College project demonstration
Hackathon/demo
Research paper
Faculty evaluation
Technical presentation

Avoid making it look like a generic food delivery website or generic chatbot.

---

49. FINAL IMPLEMENTATION REPORT

After actually implementing everything, provide a concise technical report containing:

Changes Made
Major functionality added/changed.

Architecture
Updated system architecture and data flow.

RAG Pipeline
How retrieval, ranking, context construction, and generation work.

Dataset
How USDA and Indian food datasets are integrated.

Toddler Conditions
Which toddler-specific conditions are supported and what evidence sources are used.

Dynamic Food System
How newly added foods persist.

Vector Database
How newly added/updated foods are embedded and indexed.

Recommendation Engine
How age + condition + nutrition + cuisine influence recommendations.

Visualization
Which research-quality graphs were implemented.

UI/UX
Major interface improvements.

Evaluation
Which RAG/recommendation metrics are supported.

Testing
What was tested and results.

Files Changed
Important files created/modified.

Remaining Limitations
Anything that could not be completed.

Be honest about limitations.

---

50. FINAL NON-NEGOTIABLE RULES

Do not rebuild the project unnecessarily.
Do not break existing working functionality.
Toddlers are the ONLY target population.
Do not make diabetes the focus.
Use toddler-specific evidence.
Prioritize Indian food data.
Include North Indian and South Indian cuisine.
Keep USDA as a supplementary source.
New foods must persist permanently.
New foods must become retrievable by RAG.
Prevent duplicate foods.
Do not fabricate nutritional data.
Do not fabricate medical information.
Do not fabricate citations.
Do not fabricate evaluation scores.
Do not hallucinate when evidence is insufficient.
Clearly display data provenance.
Normalize nutritional comparisons correctly.
Add publication-quality visualization.
Improve the UI substantially.
Make the entire application responsive.
Actually implement and test the changes.
