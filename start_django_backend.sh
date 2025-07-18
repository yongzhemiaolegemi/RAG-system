cd "$(dirname "$0")"
PORT=$(PYTHONPATH=. python -c "import config; print(config.django_service_port)")
python backend/manage.py runserver $PORT