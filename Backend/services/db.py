from database.mongo_connection import db


def init_db() -> None:
    """
    Initialize database connection.

    Importing `db` from `database.mongo_connection` ensures that the
    MongoDB client is created when available.
    """
    if db is None:
        return
    _ = db.list_collection_names()

