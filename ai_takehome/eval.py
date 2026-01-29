from improved_agent import answer
tests = [
    ("Why are cardiology claims denied most often?", ["Cardiology", "Coding error"]),
    ("List common denial reasons for radiology.", ["Radiology", "Invalid", "Duplicate"]),
    ("Top duplicate claim issues?", ["Duplicate"]),
    ("Why do we have expired coverage denials?", ["Expired coverage"]),
    ("What missing info causes Pediatrics denials?", ["Pediatrics", "Missing"])
]
passed = 0
for q, keywords in tests:
    ans = answer(q).lower()
    if all(k.lower() in ans for k in keywords):
        passed += 1
print(f"Score: {passed}/5")
