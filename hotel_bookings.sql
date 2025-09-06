-- This script creates a normalized database schema for the hotel booking data.
-- It separates data into multiple tables to reduce redundancy and improve data integrity.

-- Drop existing tables in reverse order of creation to avoid foreign key errors.
DROP TABLE IF EXISTS bookings;
DROP TABLE IF EXISTS guests;
DROP TABLE IF EXISTS hotels;
DROP TABLE IF EXISTS room_types;
DROP TABLE IF EXISTS countries;
DROP TABLE IF EXISTS market_segments;
DROP TABLE IF EXISTS distribution_channels;
DROP TABLE IF EXISTS deposit_types;
DROP TABLE IF EXISTS customer_types;
DROP TABLE IF EXISTS reservation_statuses;

-- Create dimension tables (the smaller lookup tables)

CREATE TABLE guests (
    guest_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    phone_number VARCHAR(50),
    credit_card VARCHAR(255)
);

CREATE TABLE hotels (
    hotel_id SERIAL PRIMARY KEY,
    hotel_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE room_types (
    room_type_id SERIAL PRIMARY KEY,
    room_type_code VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE countries (
    country_id SERIAL PRIMARY KEY,
    country_code VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE market_segments (
    segment_id SERIAL PRIMARY KEY,
    segment_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE distribution_channels (
    channel_id SERIAL PRIMARY KEY,
    channel_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE deposit_types (
    deposit_type_id SERIAL PRIMARY KEY,
    deposit_type_name VARCHAR(30) UNIQUE NOT NULL
);

CREATE TABLE customer_types (
    customer_type_id SERIAL PRIMARY KEY,
    customer_type_name VARCHAR(30) UNIQUE NOT NULL
);

CREATE TABLE reservation_statuses (
    status_id SERIAL PRIMARY KEY,
    status_name VARCHAR(30) UNIQUE NOT NULL
);


-- Create the main fact table (the central bookings table)

CREATE TABLE bookings (
    booking_id SERIAL PRIMARY KEY,
    hotel_id INT REFERENCES hotels(hotel_id),
    guest_id INT REFERENCES guests(guest_id),
    lead_time INT,
    arrival_date DATE,
    arrival_date_week_number INT,
    stays_in_weekend_nights INT,
    stays_in_week_nights INT,
    adults INT,
    children INT,
    babies INT,
    meal VARCHAR(10),
    country_id INT REFERENCES countries(country_id),
    segment_id INT REFERENCES market_segments(segment_id),
    channel_id INT REFERENCES distribution_channels(channel_id),
    is_canceled INT,
    is_repeated_guest INT,
    previous_cancellations INT,
    previous_bookings_not_canceled INT,
    reserved_room_type_id INT REFERENCES room_types(room_type_id),
    assigned_room_type_id INT REFERENCES room_types(room_type_id),
    booking_changes INT,
    deposit_type_id INT REFERENCES deposit_types(deposit_type_id),
    agent REAL, -- Using REAL as it can contain non-integer IDs
    company REAL, -- Using REAL as it can contain non-integer IDs
    days_in_waiting_list INT,
    customer_type_id INT REFERENCES customer_types(customer_type_id),
    adr REAL,
    required_car_parking_spaces INT,
    total_of_special_requests INT,
    reservation_status_id INT REFERENCES reservation_statuses(status_id),
    reservation_status_date DATE
);

-- Add indexes for frequently looked up foreign keys to improve query performance
CREATE INDEX idx_bookings_hotel_id ON bookings(hotel_id);
CREATE INDEX idx_bookings_guest_id ON bookings(guest_id);
CREATE INDEX idx_bookings_country_id ON bookings(country_id);
CREATE INDEX idx_bookings_arrival_date ON bookings(arrival_date);
