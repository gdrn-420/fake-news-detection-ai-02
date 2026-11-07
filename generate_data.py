"""Create a fictional teaching dataset for the classroom text-classification demo.

These are INVENTED examples, NOT verified news. Labels are illustrative teaching
labels: REAL = official/notice-style wording, FAKE = rumour/sensational wording.

Each sentence comes from a template (template_id). train_model.py splits the data
BY TEMPLATE, so the test set contains sentence patterns the model never saw
during training. This avoids the 'memorised the test set' problem.

Run: python generate_data.py
"""
from pathlib import Path
import csv
import random

# ---------- REAL-style templates (official notices, factual reports) ----------
REAL = [
    "The school notice says the science exhibition will take place in the {place} on {day}.",
    "The public library announced revised opening hours for the month of {month}.",
    "The district office published the schedule for the community clean-up programme in {place}.",
    "The transport department posted an update about changes to the student bus timetable.",
    "The examination board released a revised timetable on its official website.",
    "The college announced that applications for the scholarship programme are open until {date}.",
    "The weather department issued a bulletin forecasting rainfall in parts of the region on {day}.",
    "The museum website lists the opening hours and ticket information for visitors.",
    "The health department published information about a free vaccination awareness camp on {day}.",
    "The school circular asks students to submit their project files to their class teachers by {date}.",
    "The city council published the minutes of its monthly public meeting held in {month}.",
    "The university announced the dates for its upcoming entrance examinations.",
    "The railway website posted a notice about a temporary platform change for selected trains.",
    "The sports department released the fixtures for the inter-school tournament starting {date}.",
    "The public works office announced a scheduled road maintenance programme near the {place}.",
    "The science centre invited visitors to a public demonstration about renewable energy on {day}.",
    "According to the municipal office, water supply will be paused for repairs on {day}.",
    "The ministry released a report on school enrolment figures, with the data available online.",
    "Officials confirmed that the new bus route will begin operating from {date}.",
    "The hospital published its visiting hours and appointment details on the notice board.",
    "The state education department shared the guidelines for the annual day celebrations.",
    "A local newspaper reported that the bridge repair work is expected to finish in {month}.",
    "The principal informed parents about the meeting scheduled for {date} in the {place}.",
    "The post office announced new service timings for the festival season.",
    "Researchers at the university published a study on groundwater levels, citing field measurements.",
    "The election office issued a notice listing the documents required for voter registration.",
    "The agriculture department organised a training session for farmers on {day}.",
    "The company stated in its annual report that revenue grew by a small margin this year.",
]

# Hard REAL examples: real-style but contain words like 'claims', 'viral', 'rumour'
REAL_HARD = [
    "The fact-checking team reviewed a viral claim about exam cancellations and found it false, officials said.",
    "The police advised residents not to forward rumours and to wait for the official notice on {day}.",
    "A spokesperson denied the social media claim and published the correct timetable on the website.",
    "The ministry said a sensational message circulating online was incorrect and shared the verified details.",
]

# ---------- FAKE-style templates (rumours, sensational, no evidence) ----------
FAKE = [
    "A viral message claims that eating one {fruit} makes people invisible for 24 hours.",
    "An anonymous post claims that humans can fly after drinking a special {fruit} juice.",
    "A forwarded message says a secret button can change every student's exam marks instantly.",
    "A post claims that the moon will disappear on {day} and gives no evidence or source.",
    "A message promises guaranteed full marks if students share it with {number} friends.",
    "An online rumour says ordinary paper can turn into gold overnight using a secret trick.",
    "A sensational post claims that sleeping for ten minutes replaces an entire year of study.",
    "An unknown account says all school examinations are cancelled but provides no official notice.",
    "A viral claim says people can breathe underwater after eating a special leaf.",
    "A forwarded message promises instant wealth by clicking an unknown link before midnight.",
    "A post claims that a person can stop time by wearing a magnetic bracelet.",
    "An anonymous message says the city will disappear tonight but offers no reliable evidence.",
    "A sensational headline claims that one tablet gives everyone perfect memory in seconds.",
    "A viral post says a secret formula can make rocks grow into full trees overnight.",
    "A message claims that the sun will not rise on {day} and asks readers to forward it.",
    "An unknown website promises guaranteed exam success without studying or attending classes.",
    "SHOCKING: doctors do not want you to know that {fruit} cures every disease in one day!",
    "Share this with {number} people right now or your phone will stop working forever.",
    "A leaked secret says the government is hiding a miracle machine that makes free electricity.",
    "You will not believe this: a student became a millionaire in one hour using a secret app.",
    "Breaking rumour: aliens have landed near the {place} and nobody is allowed to talk about it.",
    "A WhatsApp forward says drinking hot water every hour removes all exam stress permanently.",
    "An unverified video claims that a famous cricketer can read minds after eating {fruit}.",
    "Urgent forward: a magic coin found in the {place} gives its owner everything they wish for.",
    "A post with no source says schools will close for one full year starting {day}.",
    "A viral image says a special lamp lets students learn a whole book while sleeping.",
    "A secret trick revealed: press this button {number} times to get free recharge forever.",
    "An anonymous user claims that a hidden island appeared overnight and is full of gold.",
]

# Hard FAKE examples: official-sounding wording but absurd or unsourced claims
FAKE_HARD = [
    "The department announced that gravity will be switched off for one hour on {day}, according to an unnamed source.",
    "The official notice allegedly confirms that all students will receive full marks without writing exams.",
    "The ministry reportedly published a report showing that {fruit} can replace all medicines.",
    "Officials supposedly confirmed that the school will give free gold coins to every student on {date}.",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Friday", "15 November", "next week"]
MONTHS = ["January", "March", "June", "September", "December"]
DATES = ["10 November", "15 December", "the end of this month", "next Friday"]
NUMBERS = ["5", "10", "15", "30"]
PLACES = ["auditorium", "school hall", "town square", "main library", "sports ground", "community centre"]
FRUITS = ["mango", "banana", "papaya", "guava", "coconut"]


def fill(template, rng):
    return template.format(
        day=rng.choice(DAYS), month=rng.choice(MONTHS), date=rng.choice(DATES),
        number=rng.choice(NUMBERS), place=rng.choice(PLACES), fruit=rng.choice(FRUITS),
    )


def main(per_template=6, seed=2026):
    rng = random.Random(seed)
    rows = []
    for label, templates in (("REAL", REAL + REAL_HARD), ("FAKE", FAKE + FAKE_HARD)):
        for tid, t in enumerate(templates):
            seen = set()
            tries = 0
            while len(seen) < per_template and tries < 100:
                seen.add(fill(t, rng))
                tries += 1
            for text in seen:
                rows.append((text, label, f"{label}_{tid}"))
    rng.shuffle(rows)
    out = Path("data/news_samples.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "template_id"])
        writer.writerows(rows)
    print(f"Created {len(rows)} fictional teaching examples at {out}")
    print("Reminder: examples are invented patterns, not verified real-world news.")


if __name__ == "__main__":
    main()
