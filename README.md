# Smart Closet — Backend ⚙️

The REST API and recommendation engine for **Smart Closet**, an intelligent wardrobe manager. Built with Django REST Framework, it manages users, closets, and wearables, and trains a **personalized machine-learning model per user** to recommend outfit combinations based on their feedback.

> 🔗 This is one half of a full-stack project. The web client lives in the companion repo: **[Smart-Closet-Frontend](https://github.com/NgnMsv/Smart-Closet-Frontend)** (React).

## How It Works

1. Users build **closets** and add **wearables** — each tagged with a type (shirt / pants / footwear), up to two usage categories (formal / casual / sport / general), and a dominant color stored as a hex value.
2. The **combinator** assembles candidate outfits (shirt + pants + footwear), optionally filtered by usage.
3. Users **label** combinations they like or dislike, building a personal training set.
4. A **Celery Beat** schedule periodically retrains a per-user `DecisionTreeClassifier` (`model_{user_id}.pkl` + a fitted `StandardScaler`), using each item's RGB color values and one-hot-encoded usage tags as features.
5. When a user requests an "accurate" combination, the trained model scores candidate outfits via `predict_proba` and surfaces the best match.

## Tech Stack

- **Django 5** + **Django REST Framework**
- **Authentication:** Djoser + SimpleJWT (custom email-based user model)
- **Background tasks:** Celery + django-celery-beat, with RabbitMQ as the broker
- **Machine learning:** scikit-learn (`DecisionTreeClassifier`), pandas, joblib
- **Image handling:** Cloudinary, Pillow
- **Utilities:** django-cors-headers, django-filter, python-dotenv
- **Deployment:** Docker + docker-compose

## Data Model

| Model | Description |
| --- | --- |
| `ClosetUser` | Custom user; authenticates by email (`first_name`, `last_name`, `phone_number`) |
| `Closet` | A named wardrobe belonging to a user |
| `Wearable` | A clothing item: `type`, `color` (hex), `usage_1`, `usage_2`, `image_url`, `accessible` |
| `Combination` | A shirt + pants + footwear set with a `label` (liked/disliked) for training |

## API Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /auth/users/` | Register a new user (Djoser) |
| `POST /auth/jwt/create/` | Obtain JWT access/refresh tokens |
| `/api/closets/` | CRUD for closets |
| `/api/wearables/` | CRUD for wearables |
| `/api/combinations/` | CRUD for outfit combinations + feedback |
| `GET /api/accurate-combination/<usage>/` | Get a model-scored recommendation for a usage |
| `/admin/` | Django admin |

## Getting Started

### Run with Docker (recommended)

The `docker-compose.yaml` spins up the full stack: migrations, API server, RabbitMQ, the Celery Beat scheduler, and the ML worker.

```bash
# Clone the repository
git clone https://github.com/NgnMsv/Smart-Closet-Backend.git
cd Smart-Closet-Backend

# Create your .env file (see Configuration below), then:
docker-compose up --build
```

The API will be available at [http://localhost:8000](http://localhost:8000).

### Run locally (without Docker)

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply migrations
python manage.py migrate

# Create an admin user
python manage.py createsuperuser

# Start the dev server
python manage.py runserver
```

To enable model training, run RabbitMQ and start the Celery worker and beat scheduler in separate terminals:

```bash
celery -A smart_closet worker --loglevel=info
celery -A smart_closet beat --loglevel=info
```

### Configuration

Create a `.env` file in the project root:

```env
SECRET_KEY=your_django_secret_key
DEBUG=True

# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Celery / RabbitMQ
CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
```

*(Match variable names to those read in `smart_closet/settings.py`.)*

## Project Structure

```
Smart-Closet-Backend/
├── smart_closet/        # Project config (settings, celery, urls)
├── user/                # Custom user model & auth
├── closet/              # Closet & Wearable models, views, serializers
├── combinator/          # Combination model + recommendation engine
│   ├── services/
│   │   ├── ai_services.py         # Per-user ML model (train/predict)
│   │   └── combinator_services.py # Outfit candidate generation
│   └── tasks.py         # Celery task: retrain all user models
├── sample_clothes/      # Sample item images
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

## Related

- **Web client:** [Smart-Closet-Frontend](https://github.com/NgnMsv/Smart-Closet-Frontend)

## License

Distributed under the MIT License. See `LICENSE` for details.

---

*Built with Django REST Framework · 2024*
