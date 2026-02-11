from fastapi import FastAPI, UploadFile
import pandas as pd
import uvicorn
import tempfile
import os

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/scout-mode")
async def scout_mode(players: UploadFile, performance: UploadFile):

    # Save temporary files
    with tempfile.NamedTemporaryFile(delete=False) as tmp1:
        tmp1.write(await players.read())
        players_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp2:
        tmp2.write(await performance.read())
        perf_path = tmp2.name

    # Load CSVs
    players_df = pd.read_csv(players_path)
    performance_df = pd.read_csv(perf_path)

    # Example output (later we will connect your real logic)
    result = {
        "players_uploaded": len(players_df),
        "performance_records": len(performance_df)
    }

    # Cleanup
    os.remove(players_path)
    os.remove(perf_path)

    return result


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
