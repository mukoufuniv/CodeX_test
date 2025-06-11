# Book Log Django App

This repository contains a minimal Django project for tracking the books you have read.

## Setup

1. Install dependencies (Django):
   ```bash
   pip install -r requirements.txt
   ```

2. Apply database migrations:
   ```bash
   python library/manage.py migrate
   ```

3. Start the development server:
   ```bash
   python library/manage.py runserver
   ```

Visit `http://localhost:8000/` to view the list of books. Use the Django admin at `http://localhost:8000/admin/` to add books.
