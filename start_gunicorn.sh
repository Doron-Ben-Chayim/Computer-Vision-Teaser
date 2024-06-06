#!/bin/bash
source venv/bin/activate
exec gunicorn --workers 3 --bind 127.0.0.1:8000 --log-level debug --access-logfile /home/ubuntu/Computer-Vision-Teaser/gunicorn-access.log --error-logfile /home/ubuntu/Computer-Vision-Teaser/gunicorn-error.log wsgi:app
