@echo off
cd /d %~dp0
for /f "delims=" %%p in ('python -c "import config; print(config.django_service_port)"') do set PORT=%%p
python backend\manage.py runserver %PORT%