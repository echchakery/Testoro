import os
import json

for filename in os.listdir("job_offers"):
    path = os.path.join("job_offers", filename)
    with open(path, "r") as f:
        job = json.load(f)

    if "job_title" not in job:
        job["job_title"] = "Titre non spécifié"
        with open(path, "w") as f:
            json.dump(job, f, indent=2)

print("✅ Tous les fichiers corrigés.")
