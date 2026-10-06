-- Migration: reserved "room" speaker ("Sala") for overlapping / unidentifiable speech
ALTER TABLE speakers ADD COLUMN kind TEXT NOT NULL DEFAULT 'person';
