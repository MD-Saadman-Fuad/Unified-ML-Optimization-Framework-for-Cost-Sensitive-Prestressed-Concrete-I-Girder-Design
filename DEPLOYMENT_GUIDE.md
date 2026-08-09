# Simple Deployment Guide for Non-CS Users
## How to Put the Prestressed Concrete I-Girder ML Optimizer Online

---

## 1. What Does "Deployment" Mean in Plain English?

Right now, the application runs on your personal computer (`http://localhost:8000`). Only you can see it on your screen.

**Deployment** means uploading the files to a free cloud server so that the web page gets its own **live public web address (URL)** (for example: `https://girder-ml-optimizer.onrender.com`).

Once deployed:
- Anyone with the link can open the tool on their laptop, tablet, or mobile phone.
- No Python installation is required on the user's phone or computer.
- It stays online 24/7.

---

## 2. Option 1: Easiest Free All-in-One Deployment (Recommended: Render.com)

Render is a free web hosting platform that can run your Python backend and host your Web UI together at zero cost.

### Step 1: Upload Your Code to GitHub
1. Go to [GitHub.com](https://github.com) and log in (create a free account if you don't have one).
2. Click the **+** icon in the top-right corner and select **New repository**.
3. Name it `girder-ml-optimizer` and click **Create repository**.
4. Upload all project files from your folder to this GitHub repository.

---

### Step 2: Connect GitHub to Render.com
1. Go to [Render.com](https://render.com) and create a free account using your GitHub account.
2. On your Render dashboard, click the **New +** button and select **Web Service**.
3. Choose **Build and deploy from a Git repository**, then select your `girder-ml-optimizer` repository.
4. Fill in these simple settings:
   - **Name:** `girder-ml-optimizer`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
5. Click **Create Web Service**.

---

### Step 3: Access Your Live App
- Render will spend 2–3 minutes setting up your app.
- Once finished, Render will display a public link at the top (e.g. `https://girder-ml-optimizer.onrender.com`).
- Open `https://girder-ml-optimizer.onrender.com` in your browser (it automatically redirects to `/ui/index.html`).
- **Congratulations! Your ML Bridge Optimizer is now live on the internet!** 🎉

---

## 3. Option 2: Deploying Separate Frontend & Backend

If you want your website to have a custom domain (e.g., `https://girder-optimizer.vercel.app`), you can host the frontend on **Vercel** or **Netlify** and the backend on **Render**.

### Step 1: Deploy Backend on Render
Deploy the API on Render as described in Option 1 above. Note down your backend URL (e.g. `https://girder-ml-api.onrender.com`).

### Step 2: Point the Frontend to Your Live API
Open `ui/app.js` on line 3 and change `API_BASE_URL`:
```javascript
// BEFORE (Localhost):
const API_BASE_URL = "http://localhost:8000";

// AFTER (Live Render API):
const API_BASE_URL = "https://girder-ml-api.onrender.com";
```

### Step 3: Deploy Frontend on Vercel
1. Go to [Vercel.com](https://vercel.com) and sign in with GitHub.
2. Click **Add New...** -> **Project**.
3. Select your `girder-ml-optimizer` repository.
4. Set **Root Directory** to `ui`.
5. Click **Deploy**.
6. You get a public address like `https://girder-ml-optimizer.vercel.app`.

---

## 4. Option 3: Docker Deployment (For IT Departments & Private Servers)

If a university or engineering company wants to host this tool on their private internal server, we have included ready-to-use Docker configuration files (`Dockerfile` and `docker-compose.yml`).

### How to Run on Any Server with Docker Installed:
Open terminal on the server and run:
```bash
docker compose up -d
```
That's it! The system will build and run automatically on port 8000.

---

## 5. Summary Checklist Before Sharing Your Link

- [x] `models/scaler.pkl` and `models/best_model.pkl` are committed in the project repository.
- [x] `requirements.txt` contains all pinned libraries.
- [x] CORS middleware in `api/main.py` is enabled for public access (`allow_origins=["*"]`).
- [x] Fallback client-side solver in `ui/app.js` handles 502 Bad Gateway cold-starts gracefully so the UI never displays errors.`n- [x] Dynamic `window.location.origin` in `ui/app.js` eliminates CORS issues across all deployment domains.`n- [x] Root `GET /` endpoint in `api/main.py` redirects to `/ui/index.html` automatically.`n- [x] `api/main.py` reads `os.getenv("PORT", 8000)` for dynamic cloud port binding.

---

## 6. How Users Experience the Deployed Application

When an engineer opens your live URL:
1. They enter **Concrete cost** ($/yd³), **Strand cost** ($/ft), **Rebar cost** ($/lb), and **Span length** (ft).
2. The trained surrogate model calculates the optimal design parameters in **milliseconds**.
3. The **SVG canvas** draws the exact cross-section of the I-girder with tendon strands live on their screen.
4. They can click **Export Report** to download the calculated parameters as a CSV file.
