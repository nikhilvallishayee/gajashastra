-- Seed data for Gajashastra
-- Run after migration.sql

INSERT INTO texts (id, title, title_devanagari, author, author_devanagari, language, description, source_url, total_pages, metadata_json, created_at, updated_at)
VALUES ('03048073-d59b-4753-acbb-89f750689415', 'Gajashastra', 'गजशास्त्रम्', 'Palakapya Muni', 'पालकाप्य मुनि', 'sanskrit', 'Ancient Sanskrit treatise on elephant science by sage Palakapya. Saraswati Mahal Series No. 76 (1958)', 'https://archive.org/details/Gajasasthra', 457, '{''year'': 1958, ''edition'': ''Saraswati Mahal Series No. 76''}', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('4af6f0ff-2ac2-4b60-946a-cffc24e934f3', '03048073-d59b-4753-acbb-89f750689415', 'Upodghata', 'उपोद्धातः', 1, 1, 40, 'Chapter 1: Upodghata', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('25296281-e419-4287-b2cd-20b7bf58ba12', '03048073-d59b-4753-acbb-89f750689415', 'Palakapya Charitram', 'पालकाप्यचरित्रम्', 2, 41, 80, 'Chapter 2: Palakapya Charitram', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('d2524cae-7283-46d4-872e-225e8ed794bd', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Lakshanam', 'गजलक्षणम्', 3, 81, 140, 'Chapter 3: Gaja Lakshanam', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('94e76d8d-40bf-4d62-ad1f-129357bda947', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Grahanam', 'गजग्रहणम्', 4, 141, 180, 'Chapter 4: Gaja Grahanam', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('44d8e444-1c74-45a6-9874-850e372f9cce', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Shiksha', 'गजशिक्षा', 5, 181, 230, 'Chapter 5: Gaja Shiksha', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('ee8dd885-65ad-4a8c-9ba7-f18757a282da', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Chikitsa', 'गजचिकित्सा', 6, 231, 340, 'Chapter 6: Gaja Chikitsa', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('b2352e6f-3ea6-40e5-a3b0-b074e26b7e73', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Ahara', 'गजाहारः', 7, 341, 390, 'Chapter 7: Gaja Ahara', NOW(), NOW());
INSERT INTO chapters (id, text_id, title, title_devanagari, "order", page_start, page_end, summary, created_at, updated_at)
VALUES ('ea7f70b4-00a6-44d4-946b-9efa778449fc', '03048073-d59b-4753-acbb-89f750689415', 'Gaja Prayoga', 'गजप्रयोगः', 8, 391, 457, 'Chapter 8: Gaja Prayoga', NOW(), NOW());

-- 39 verses
-- 39 embeddings
-- Verse and embedding INSERT statements omitted for brevity
-- Use the dataset-export.json with the seed script instead