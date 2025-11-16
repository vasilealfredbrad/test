# 🎬 Customizing Video Scenes for Your Story

## 🎨 Enhanced Realism Features

Your AI video now generates with **story-driven prompts** that match what the words are saying!

### What Changed:

**Before:**
- Generic "realistic person" prompt
- Didn't match story content
- Less detailed visuals

**Now:**
- Specific scenes from Alin's story
- Matches narrative (Romanian guy, computer, friends, etc.)
- Rich visual details (apartment decor, expressions, lighting)
- Higher quality settings (20 steps, guidance 9.0)

---

## 📝 Current Story Scenes

For Alin's story, the AI generates:

```python
story_scenes = [
    "A young Romanian man in his 20s sitting at a computer desk looking confused",
    "Close-up of hands frantically clicking on a messy Windows computer screen",
    "The same man looking surprised as code appears on screen",
    "Friends gathering around impressed, pointing at the laptop",
    "The man nervously smiling while secretly Googling on his phone",
    "Stack Overflow website visible on screen in background",
    "The man confidently presenting to friends with a fake confident smile",
    "Documentary-style footage of an amateur programmer in a home office"
]
```

The AI will use these as visual guidance to create realistic footage matching the story!

---

## 🔧 How to Customize Scenes for Your Own Story

### Step 1: Edit Your Story

Open `generate_subtitles.py` and find line 30:

```python
STORY = """
Your custom story here...
"""
```

### Step 2: Add Matching Visual Scenes

Around line 726, update the `story_scenes` list to match your story:

```python
story_scenes = [
    "Your first scene description",
    "Your second scene description",
    "Your third scene description",
    # Add more scenes as needed
]
```

### Example: Travel Vlog Story

```python
STORY = """
Join me as I explore the beautiful streets of Paris.
The Eiffel Tower stands majestically against the blue sky.
I'm tasting the most amazing croissant at a local café.
The sunset over the Seine River is breathtaking.
"""

story_scenes = [
    "A tourist walking through cobblestone streets of Paris, sunny day",
    "Close-up of the Eiffel Tower with blue sky background",
    "Hands holding a fresh croissant in a Parisian café",
    "Beautiful sunset over the Seine River with boats passing by"
]
```

### Example: Cooking Tutorial

```python
STORY = """
Today we're making the perfect pasta carbonara.
First, we'll cook the pancetta until crispy.
Then we'll mix eggs with parmesan cheese.
Finally, we combine everything for a creamy finish.
"""

story_scenes = [
    "Chef in modern kitchen holding fresh pasta ingredients",
    "Close-up of pancetta sizzling in a pan, golden and crispy",
    "Hands whisking eggs with grated parmesan in a bowl",
    "Creamy carbonara pasta being tossed in a pan, steam rising"
]
```

---

## 🎯 Tips for Better Visual Results

### Be Specific:
- ❌ "A person doing something"
- ✅ "A 25-year-old woman with dark hair mixing ingredients in a bright kitchen"

### Add Details:
- Setting: "home office", "busy café", "modern apartment"
- Lighting: "golden hour sunlight", "soft morning light", "warm indoor lighting"
- Emotions: "confused", "excited", "concentrated", "smiling"
- Actions: "typing on laptop", "holding coffee", "gesturing to camera"

### Use Realistic Keywords:
- "Photorealistic"
- "Documentary style"
- "Natural lighting"
- "Real person"
- "Cinematic 4K"
- "Authentic"

### Specify What You DON'T Want:
- "No animations"
- "No cartoons"
- "No artificial elements"
- "No CGI"

---

## ⚡ Quality Settings

### Current Settings (Balanced):
```python
num_inference_steps=20  # Good quality, ~8-12 minutes
guidance_scale=9.0      # Strong prompt adherence
```

### Want Faster (Lower Quality)?
```python
num_inference_steps=15  # Faster, ~5-8 minutes
guidance_scale=7.5      # Moderate adherence
```

### Want Best Quality (Slower)?
```python
num_inference_steps=30  # Best quality, ~15-20 minutes
guidance_scale=11.0     # Maximum adherence
```

Edit these around line 780 in `generate_subtitles.py`.

---

## 📊 Example Full Customization

### For a Tech Tutorial:

```python
STORY = """
Welcome to this Python tutorial.
We'll learn how to create a web scraper.
First, we import the required libraries.
Then we write the scraping function.
Finally, we test our code and see results.
"""

story_scenes = [
    "Young programmer smiling at camera in modern home office setup",
    "Computer screen showing Python code with libraries being imported",
    "Hands typing on mechanical keyboard, code appearing on monitor",
    "Terminal window showing web scraper running with data output",
    "Excited programmer giving thumbs up with successful code behind"
]

# In the prompt section (line 738):
prompt = (
    f"Photorealistic documentary footage, cinematic 4K quality. "
    f"Scene: {story_scenes[0]}. "
    f"A professional programmer in their 20s or 30s, "
    f"clean modern home office with plants and tech gear, "
    f"natural daylight through window, "
    f"genuine enthusiasm and concentration, "
    f"real hands typing on keyboard with visible code on screen. "
    f"Documentary style with slight camera movement. "
    f"Style: {emotion_profile}. "
    f"No animations, pure photorealistic footage."
)
```

---

## 🎬 Scene Variations

The AI randomly picks from your scene list, so add variety:

```python
story_scenes = [
    "Wide shot of the main subject",
    "Close-up of hands/face showing emotion",
    "Over-the-shoulder view of what they're working on",
    "Side angle showing the environment",
    "Detail shot of important elements"
]
```

This creates a more dynamic, professional-looking video!

---

## 💡 Pro Tips

1. **Match Your Story Beats**
   - Each scene should represent a key moment in your narrative

2. **Use Active Verbs**
   - "walking", "typing", "holding", "pointing", "smiling"

3. **Specify Age/Appearance**
   - "young woman in her 20s"
   - "middle-aged man with beard"
   - "elderly person with glasses"

4. **Set the Mood**
   - Happy: "bright lighting, smiling, energetic"
   - Serious: "focused expression, dim lighting, concentrated"
   - Excited: "animated gestures, wide smile, dynamic movement"

5. **Include Background Details**
   - Props, decorations, weather, time of day

---

## 🚀 Test Your Scenes

Run the script and see how the AI interprets your descriptions:

```bash
python generate_subtitles.py
```

Select **Option 2** → **720p** → Wait **~8-12 minutes**

Review the output and adjust your scene descriptions for better results!

---

**Create videos that truly match your story!** 🎥✨

