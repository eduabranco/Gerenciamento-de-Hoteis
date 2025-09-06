import psycopg2
import csv
from datetime import datetime
import sys

# --- Database Connection Details ---
# Replace with your actual PostgreSQL connection details
DB_NAME = "your_db_name"
DB_USER = "your_db_user"
DB_PASS = "your_db_password"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- CSV File Details ---
CSV_FILE_PATH = 'data/hotel_booking.csv'

# In-memory caches to store foreign keys and avoid repeated DB lookups
# Format: { 'value_from_csv': primary_key_in_db }
dimension_caches = {
    'guests': {},
    'hotels': {},
    'room_types': {},
    'countries': {},
    'market_segments': {},
    'distribution_channels': {},
    'deposit_types': {},
    'customer_types': {},
    'reservation_statuses': {}
}

# Month name to number mapping
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def get_or_create_dimension_id(cursor, table_name, key_column, key_value, extra_cols=None):
    """
    Generic function to get the ID for a given value from a dimension table.
    If the value doesn't exist, it inserts it and returns the new ID.
    Uses an in-memory cache to reduce database queries.
    """
    cache_key = f"{table_name}_{key_value}"
    cache = dimension_caches.get(table_name.split('(')[0].strip())

    if cache is not None and key_value in cache:
        return cache[key_value]

    # Handle NULL-like values from CSV
    if key_value is None or key_value.strip() == '' or key_value.upper() == 'NULL' or key_value.upper() == 'UNDEFINED':
        return None

    try:
        # Check database if not in cache
        cursor.execute(f"SELECT {table_name.split('(')[0]}_id FROM {table_name} WHERE {key_column} = %s", (key_value,))
        result = cursor.fetchone()

        if result:
            entity_id = result[0]
            if cache is not None:
                cache[key_value] = entity_id
            return entity_id
        else:
            # Insert if not found
            if extra_cols:
                cols = [key_column] + list(extra_cols.keys())
                vals = [key_value] + list(extra_cols.values())
                placeholders = ', '.join(['%s'] * len(cols))
                sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders}) RETURNING {table_name.split('(')[0]}_id;"
                cursor.execute(sql, vals)
            else:
                sql = f"INSERT INTO {table_name} ({key_column}) VALUES (%s) RETURNING {table_name.split('(')[0]}_id;"
                cursor.execute(sql, (key_value,))

            entity_id = cursor.fetchone()[0]
            if cache is not None:
                cache[key_value] = entity_id
            return entity_id

    except psycopg2.Error as e:
        print(f"Error handling dimension '{table_name}' for value '{key_value}': {e}")
        # Return None to allow the main loop to continue and potentially rollback
        return None

def load_data():
    """
    Loads data from CSV into a normalized PostgreSQL schema.
    """
    conn = None
    record_count = 0
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()
        print("Successfully connected to the database.")

        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    # Clean up phone number column name if it has extra characters
                    row['phone-number'] = row.get('phone-number', row.get('\ufeffphone-number'))

                    # 1. Get IDs for all dimension tables
                    guest_details = {
                        'name': row['name'],
                        'phone_number': row['phone-number'],
                        'credit_card': row['credit_card']
                    }
                    guest_id = get_or_create_dimension_id(cursor, 'guests', 'email', row['email'], guest_details)

                    hotel_id = get_or_create_dimension_id(cursor, 'hotels', 'hotel_name', row['hotel'])
                    country_id = get_or_create_dimension_id(cursor, 'countries', 'country_code', row['country'])
                    segment_id = get_or_create_dimension_id(cursor, 'market_segments', 'segment_name', row['market_segment'])
                    channel_id = get_or_create_dimension_id(cursor, 'distribution_channels', 'channel_name', row['distribution_channel'])
                    deposit_type_id = get_or_create_dimension_id(cursor, 'deposit_types', 'deposit_type_name', row['deposit_type'])
                    customer_type_id = get_or_create_dimension_id(cursor, 'customer_types', 'customer_type_name', row['customer_type'])
                    status_id = get_or_create_dimension_id(cursor, 'reservation_statuses', 'status_name', row['reservation_status'])

                    # Room types can appear in two columns, ensure they are in the table
                    reserved_room_type_id = get_or_create_dimension_id(cursor, 'room_types', 'room_type_code', row['reserved_room_type'])
                    assigned_room_type_id = get_or_create_dimension_id(cursor, 'room_types', 'room_type_code', row['assigned_room_type'])

                    # 2. Prepare data for the bookings table
                    arrival_date = datetime(
                        int(row['arrival_date_year']),
                        MONTH_MAP[row['arrival_date_month']],
                        int(row['arrival_date_day_of_month'])
                    ).date()
                    
                    # Convert empty strings for numeric types to None (for NULL)
                    agent_id = float(row['agent']) if row['agent'] and row['agent'] != 'NULL' else None
                    company_id = float(row['company']) if row['company'] and row['company'] != 'NULL' else None
                    children_count = int(float(row['children'])) if row['children'] and row['children'] != 'NULL' else 0


                    # 3. Insert into the main bookings table
                    insert_sql = """
                        INSERT INTO bookings (
                            hotel_id, guest_id, lead_time, arrival_date, arrival_date_week_number,
                            stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal,
                            country_id, segment_id, channel_id, is_canceled, is_repeated_guest,
                            previous_cancellations, previous_bookings_not_canceled,
                            reserved_room_type_id, assigned_room_type_id, booking_changes,
                            deposit_type_id, agent, company, days_in_waiting_list, customer_type_id,
                            adr, required_car_parking_spaces, total_of_special_requests,
                            reservation_status_id, reservation_status_date
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    cursor.execute(insert_sql, (
                        hotel_id, guest_id, int(row['lead_time']), arrival_date, int(row['arrival_date_week_number']),
                        int(row['stays_in_weekend_nights']), int(row['stays_in_week_nights']), int(row['adults']),
                        children_count, int(row['babies']), row['meal'], country_id, segment_id, channel_id,
                        int(row['is_canceled']), int(row['is_repeated_guest']), int(row['previous_cancellations']),
                        int(row['previous_bookings_not_canceled']), reserved_room_type_id, assigned_room_type_id,
                        int(row['booking_changes']), deposit_type_id, agent_id, company_id,
                        int(row['days_in_waiting_list']), customer_type_id, float(row['adr']),
                        int(row['required_car_parking_spaces']), int(row['total_of_special_requests']),
                        status_id, row['reservation_status_date']
                    ))
                    record_count += 1
                    # Provide progress feedback
                    if (i + 1) % 1000 == 0:
                        sys.stdout.write(f"\rProcessed {i+1} records...")
                        sys.stdout.flush()
                        conn.commit() # Commit in batches

                except (ValueError, KeyError) as e:
                    print(f"\nSkipping row {i+1} due to data error: {e}. Row data: {row}")
                    conn.rollback() # Rollback the failed transaction for this row
                except psycopg2.Error as e:
                    print(f"\nSkipping row {i+1} due to database error: {e}. Row data: {row}")
                    conn.rollback()

        conn.commit() # Final commit
        print(f"\n\nData loading complete. Total records inserted: {record_count}")

    except psycopg2.Error as e:
        print(f"\nDatabase connection error: {e}")
    except FileNotFoundError:
        print(f"\nError: The file {CSV_FILE_PATH} was not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    load_data()
