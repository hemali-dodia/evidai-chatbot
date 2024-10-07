@echo off
waitress-serve --port=8000 evidai.wsgi:application
pause