import json
import random

# --- ESOTERIC KNOWLEDGE BASE (Public Domain / Waite-Smith Inspired) ---

MAJOR_ARCANA = {
    "The Fool": {"u": "pure potential, innocent leaps of faith, and unburdened beginnings", "r": "reckless folly, naivety, and hesitations before the precipice"},
    "The Magician": {"u": "manifestation, the harnessing of elemental forces, and willful creation", "r": "manipulation, untapped potential, and illusions of control"},
    "The High Priestess": {"u": "intuition, the lifting of the veil, and the whispers of the subconscious", "r": "hidden agendas, blocked psychic reception, and ignored instincts"},
    "The Empress": {"u": "abundant nurturing, maternal fertility, and the flourishing of nature", "r": "creative stagnation, domestic overbearance, and neglected self-care"},
    "The Emperor": {"u": "structured authority, paternal foundation, and the establishment of order", "r": "tyrannical control, rigid inflexibility, and collapsing structures"},
    "The Hierophant": {"u": "traditional wisdom, spiritual conformity, and the seeking of orthodox counsel", "r": "rebellion against dogma, unorthodox beliefs, and spiritual restriction"},
    "The Lovers": {"u": "divine union, harmonious choices, and the alignment of values", "r": "disharmony, fractured relationships, and choices made from fear"},
    "The Chariot": {"u": "triumphant willpower, focused momentum, and the conquering of opposing forces", "r": "loss of direction, aggressive impulses, and stalled progress"},
    "Strength": {"u": "gentle fortitude, the taming of the inner beast, and compassionate courage", "r": "raw aggression, self-doubt, and the triumph of base instincts"},
    "The Hermit": {"u": "introspective solitude, inner guidance, and the seeking of the inner light", "r": "forced isolation, loneliness, and withdrawal from the world"},
    "Wheel of Fortune": {"u": "inevitable cycles, the turning of fate, and karmic destiny", "r": "bad luck, clinging to control, and resisting the turning tide"},
    "Justice": {"u": "karmic balance, objective truth, and the weighing of the scales", "r": "unfairness, dishonesty, and the avoidance of accountability"},
    "The Hanged Man": {"u": "sacrifice for wisdom, suspended perspectives, and spiritual surrender", "r": "needless martyrdom, stalling, and resistance to surrender"},
    "Death": {"u": "necessary endings, profound transformation, and the clearing of the old", "r": "fear of change, painful stagnation, and clinging to decaying forms"},
    "Temperance": {"u": "alchemical alchemy, moderate blending, and the finding of the middle path", "r": "extremes, destructive imbalance, and volatile combinations"},
    "The Devil": {"u": "shadow bonds, material addictions, and the illusion of entrapment", "r": "release from chains, facing the shadow, and breaking toxic cycles"},
    "The Tower": {"u": "sudden upheaval, the shattering of false foundations, and chaotic revelation", "r": "delaying the inevitable crash, fear of disaster, and avoiding truth"},
    "The Star": {"u": "renewed hope, celestial healing, and the serene pouring of cosmic waters", "r": "despair, disconnected faith, and a lack of spiritual inspiration"},
    "The Moon": {"u": "subconscious illusions, poetic confusion, and the journey through the psychic night", "r": "release of fear, revealing of secrets, and overcoming anxiety"},
    "The Sun": {"u": "radiant vitality, joyful illumination, and the clarity of the waking day", "r": "temporary sadness, blocked joy, and a lack of enthusiasm"},
    "Judgement": {"u": "ultimate rebirth, the hearing of the cosmic call, and rising to a higher self", "r": "ignoring the call, self-doubt, and remaining in the karmic grave"},
    "The World": {"u": "complete fulfillment, the dancing of the cosmic cycle, and ultimate harmony", "r": "incomplete cycles, delayed closure, and a lack of finality"}
}

SUITS = {
    "Wands": {"element": "Fire", "domain": "passion, willpower, creative sparks, and the forging of destiny"},
    "Cups": {"element": "Water", "domain": "the depths of emotion, intuition, relationships, and the vessel of the soul"},
    "Swords": {"element": "Air", "domain": "the piercing intellect, conflict, truth, and the realm of thoughts and words"},
    "Pentacles": {"element": "Earth", "domain": "material foundation, worldly harvest, physical bodies, and ancestral roots"}
}

NUMEROLOGY = {
    "Ace": "the pure, unmanifested seed",
    "Two": "the duality of choice and partnership",
    "Three": "the first fruits of creation and collaboration",
    "Four": "the establishment of stability and rigid structure",
    "Five": "the necessary chaos and conflict of growth",
    "Six": "the restoration of harmony and temporary victory",
    "Seven": "the inward reflection and mystical assessment",
    "Eight": "the mastery of movement and deliberate action",
    "Nine": "the near-completion and the gathering of final strength",
    "Ten": "the ultimate culmination, be it a heavy burden or a joyous zenith",
    "Page": "the youthful messenger and eager student",
    "Knight": "the extreme action and dedicated pursuit",
    "Queen": "the internal mastery and nurturing embodiment",
    "King": "the external authority and commanded rulership"
}

# --- GENERATOR LOGIC ---

INTENTS = [
    "Will I find love?",
    "Is this career path right for me?",
    "What do I need to know about my spiritual journey?",
    "General Reading",
    "How can I overcome my current obstacles?",
    "What is the hidden truth of this situation?",
    "Where is my creative energy leading me?",
    "Should I hold on or let go?"
]

OPENINGS = [
    "Hark, seeker of the veiled truth. The ether trembles as your intent is cast upon the cosmic loom.",
    "Child of the turning wheel, the Oracle hears your plea. The mists part to reveal the architecture of your fate.",
    "Welcome, traveler. You stand at the crossroads, and the cards have gathered to whisper the secrets of the abyss.",
    "The query is spoken, and the ancient archetypes awaken. Let us read the geometry of your soul.",
    "Step into the sacred circle, Querent. The shadows dance and the archetypes assemble to answer your call."
]

def generate_minor_arcana():
    suit = random.choice(list(SUITS.keys()))
    num = random.choice(list(NUMEROLOGY.keys()))
    card_name = f"{num} of {suit}"
    
    is_reversed = random.random() > 0.7
    
    meaning = f"the energy of {SUITS[suit]['element']}, governing {SUITS[suit]['domain']}"
    num_meaning = NUMEROLOGY[num]
    
    if is_reversed:
        insight = f"a blockage or internalization of {num_meaning} within {meaning}"
    else:
        insight = f"the flowing manifestation of {num_meaning} within {meaning}"
        
    return card_name, is_reversed, insight

def generate_major_arcana():
    card_name = random.choice(list(MAJOR_ARCANA.keys()))
    is_reversed = random.random() > 0.7
    state = "r" if is_reversed else "u"
    insight = MAJOR_ARCANA[card_name][state]
    return card_name, is_reversed, insight

def generate_reading():
    intent = random.choice(INTENTS)
    spread_type = random.choice(["Single Card", "Three Card (Past, Present, Future)"])
    
    cards_drawn = []
    insights = []
    
    num_cards = 1 if spread_type == "Single Card" else 3
    
    for i in range(num_cards):
        if random.random() > 0.3:
            name, rev, ins = generate_minor_arcana()
        else:
            name, rev, ins = generate_major_arcana()
            
        pos = "Single" if num_cards == 1 else ["Past", "Present", "Future"][i]
        cards_drawn.append({"name": name, "isReversed": rev, "position": pos})
        insights.append((name, rev, ins, pos))
        
    # Construct User Prompt
    cards_str = ", ".join([f"{c['name']} ({'Reversed' if c['isReversed'] else 'Upright'})" for c in cards_drawn])
    user_input = f"Spread: {spread_type}\\nIntent: {intent}\\nCards: {cards_str}"
    
    # Construct Oracle Response
    opening = random.choice(OPENINGS)
    
    body = ""
    for name, rev, ins, pos in insights:
        rev_str = "shadowed and reversed" if rev else "standing tall and upright"
        if pos == "Single":
            body += f"\\n\\nFrom the deck emerges **{name}**, {rev_str}. It speaks to you of {ins}. This singular vision demands your full attention, answering your query with the absolute truth of its archetype."
        else:
            if pos == "Past":
                body += f"\\n\\nIn the foundation of your **Past**, we see **{name}**, {rev_str}. This signifies that the roots of your current state were forged in {ins}."
            elif pos == "Present":
                body += f"\\n\\nIn the crucible of your **Present**, **{name}** reveals itself, {rev_str}. You are currently swimming in {ins}. This is the fulcrum of your choices."
            elif pos == "Future":
                body += f"\\n\\nLooking toward the unfolding **Future**, the winds blow toward **{name}**, {rev_str}. If you remain on this path, you will encounter {ins}."

    closing = "\\n\\nThe threads are woven, Querent. Meditate upon these symbols, for the true magic lies in your own reflection."
    
    oracle_output = opening + body + closing
    
    return {
        "contents": [
            {"role": "user", "parts": [{"text": user_input}]},
            {"role": "model", "parts": [{"text": oracle_output}]}
        ]
    }

print("Generating 1000 mystical readings...")
dataset = []
for _ in range(1000):
    dataset.append(generate_reading())

with open("tarot_oracle_training.jsonl", "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\\n")

print("Successfully generated tarot_oracle_training.jsonl")
