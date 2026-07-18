from database.mongo_connection import db


def init_db() -> None:
    """
    Initialize database connection.

    Importing `db` from `database.mongo_connection` ensures that the
    MongoDB client is created and the application can access the database.
    """

    # Trigger a lightweight operation to verify the connection is usable.
    # This is intentionally minimal to avoid heavy startup work.
    _ = db.list_collection_names()

