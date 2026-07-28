This is actually a very realistic Document AI problem, not an OCR problem.

Most people will fail because they'll think:

"Run OCR on every image."

But that's not what Aaditya is testing.

He's testing whether you think like someone building an actual production system where every API call costs money.

First understand the dataset

Look at your images.

Category 1 — Printed documents

These contain structured information.

Examples

Exide Invoice
Tyre service bill
DEF meter receipt
Printed work report

These should produce JSON.

Example

{
  "document_type":"invoice",
  "vendor":"Anupam Enterprise",
  "invoice_no":"AE2638/26-27",
  "date":"20-06-2026",
  "customer":"Avtarsingh...",
  "items":[
      {
         "name":"XP1000 Battery",
         "quantity":2,
         "amount":13220.34
      }
  ],
  "total":15600
}
Category 2 — Handwritten documents

Examples

Mechanic notebook
Work report
Depot register

Still documents.

Need OCR.

Need JSON.

Example

{
    "document_type":"mechanic_log",
    "date":"17-06-2026",
    "entries":[
        {
            "vehicle":"TCM40",
            "work":"Oil change"
        },
        {
            "vehicle":"TCM19",
            "work":"Brake repair"
        }
    ]
}
Category 3 — Meter / Dashboard

Example

DEF machine

Need

{
    "document_type":"meter_reading",
    "amount":3199.76,
    "litres":43.24,
    "price_per_litre":74,
    "urea":32.5
}
Category 4 — NON DOCUMENTS

These are the trick.

Battery

Tyre

Truck battery

Tube

Random object

These should NOT go to OCR.

Instead

{
    "document_type":"non_document",
    "reason":"vehicle battery"
}

or

{
    "document_type":"non_document",
    "reason":"tyre"
}

Notice

You should NOT hallucinate

No OCR.

No invoice.

No rows.

No fake JSON.

This is what Aaditya means

He literally wrote

refuse the ones that aren't (don't invent rows from a battery photo)

Meaning

Battery image →

Return

{
   "status":"rejected",
   "reason":"Non-document image"
}

Done.

So pipeline should look like this
                Image
                  │
                  ▼
      Document / Non-document classifier
            │                 │
            │                 │
         Document         Non-document
            │                 │
            ▼                 ▼
      OCR Pipeline      Reject immediately
            │
            ▼
   Document Type Detection
            │
 ┌──────────┼───────────────┐
 │          │               │
Invoice   Handwritten    Meter
 │          │               │
 ▼          ▼               ▼
LLM      OCR+LLM        Small OCR
 │
 ▼
Structured JSON

That is exactly how a real document AI company works.

Now the cost optimisation

This is probably 60% of the assignment.

He doesn't want

Every image
↓

GPT-4.1 Vision
↓

JSON

Because

12 images

↓

12 expensive API calls.

Instead

Step 1

Use OpenCV

No AI.

Free.

Detect

edges
text density
contours

If almost no text

Reject.

Battery images never reach GPT.

Cost

£0.

Step 2

Use Tesseract OCR

Free.

Run OCR.

If OCR returns

Invoice

GST

Amount

Date

Probably invoice.

No GPT yet.

If OCR returns

TCM35

Oil

Mechanic


Probably handwritten.

If OCR returns

3199

43.24


Probably meter.

Step 3

Now call small model

Like

GPT-4.1 nano

Gemini Flash

Claude Haiku

instead of GPT-4.1

Only for parsing.

Prompt

Convert OCR text into JSON.

Do not hallucinate.

If field missing return null.

Very cheap.

Step 4

Only difficult documents

Bad handwriting

Poor scan

Mixed language

Send to GPT-4.1

Maybe only

10%

of images.

Huge savings.

Baseline
12 images

↓

GPT-4.1 Vision

↓

JSON

Cost

100%

Optimized
12 images

↓

OpenCV classifier

↓

4 rejected

↓

8 OCR

↓

6 parsed by Nano

↓

2 difficult

↓

GPT-4.1

Cost

Maybe

20–30%

of baseline.

Accuracy almost identical.

That's exactly what they're looking for.

How to detect a non-document?

You don't even need an LLM.

Use OCR confidence.

Example

Run Tesseract.

If

Detected text

""


or

3 words

Probably not document.

Battery image

↓

Reject.

Invoice

↓

500 words

↓

Process.

Simple.

Tech stack I'd use
Python

OpenCV
↓
Tesseract OCR
↓
EasyOCR (fallback)
↓
GPT-4.1 Nano / Gemini Flash
↓
Pydantic JSON schema
↓
Cost logger
Suggested project structure
optera-doc-ai/

│
├── input/
├── output/
│
├── classify.py
├── preprocess.py
├── ocr.py
├── router.py
├── extract.py
├── schemas.py
├── cost_logger.py
├── main.py
│
├── DESIGN.md
├── README.md
└── requirements.txt

Run everything with:

python main.py --input input/
What I would implement in 24 hours
Baseline: Every image → GPT-4.1 Vision → JSON.
Optimized pipeline:
OpenCV + OCR to filter non-documents.
Route printed documents to Tesseract + GPT-4.1 Nano.
Route handwritten pages to EasyOCR + GPT-4.1 Nano (or GPT-4.1 only if OCR confidence is low).
Route meter displays to a lightweight OCR path.
Reject batteries, tyres and other non-documents with a structured reason.
Metrics:
Log every model call, input/output tokens, estimated cost, processing time and routing decision.
Compare baseline vs optimized on accuracy and cost.

This demonstrates not only OCR skills but also production thinking—routing, validation, and cost optimisation—which is likely the main goal of the challenge.

If you want to maximize your chances, I can also help you build the entire repository (end-to-end code, prompts, JSON schemas, cost logger, and DESIGN.md) so it looks like a professional submission.

so for first part i have to check like image is invoice or tyre? so for that i have to set llm or with coding it is possible?

Exactly! This is the most important design decision in the assignment.

The answer is don't use an LLM for the first step if your goal is to minimize cost. Use code first.

Think like this:

Image
   │
   ▼
Document Detector (FREE)
   │
   ├── Non-document → Reject
   │
   └── Document
          │
          ▼
      OCR + LLM
Option 1 (Recommended): Pure Python/OpenCV + OCR

You can detect whether it's a document without any LLM.

1. Check text amount

Run Tesseract or EasyOCR.

Example:

import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open("image.jpg"))

if len(text.strip()) < 20:
    print("Probably not a document")
else:
    print("Document")

Your battery image will probably return:

EXIDE
INVA TEC

Only 2–3 words.

Invoice returns hundreds of words.

Notebook returns many words.

2. OCR confidence

EasyOCR returns confidence.

Example

results = reader.readtext(image)

avg_conf = sum(r[2] for r in results)/len(results)

If

confidence very low
very few detected text boxes

↓

Reject.

3. Detect paper boundary

Invoices usually have

rectangle
white background
many horizontal lines

Battery image

no page
no table
no document layout

OpenCV can detect this.

Option 2: Small Vision Model (Better Accuracy)

If you don't mind spending a tiny amount, use a very cheap model only for routing.

Example prompt:

Classify this image into ONE category:

- invoice
- receipt
- handwritten_log
- meter_reading
- non_document

Return JSON only.

Use:

GPT-4.1 Nano
Gemini Flash
Claude Haiku

Cost is extremely low.

Then:

Battery
↓

{
 "type":"non_document"
}

↓

Stop.
What would I do?

I'd combine both.

Image
   │
   ▼
OCR
   │
   ├── No text
   │       │
   │       ▼
   │   Non-document
   │
   ▼
Enough text
   │
   ▼
Small LLM
   │
   ├── Invoice
   ├── Handwritten
   ├── Meter
   └── Other

This is much cheaper than sending every image to GPT-4.1.

Let's test it on your images
Image	OCR?	Route
Exide invoice	Lots of text	Invoice parser
Tyre service receipt	Lots of text	Receipt parser
DEF meter	Numbers + labels	Meter parser
Handwritten notebook	Lots of text	Handwritten parser
Work report	Lots of text	Handwritten parser
Battery photo	Very little text	Reject
Truck tyre	Almost no text	Reject
Battery installed in truck	Very little text	Reject
Tube photo	No meaningful text	Reject
What would impress the interviewer?

Don't write:

"I used GPT-4.1 Vision on every image."

Write something like:

"To minimize cost, I implemented a routing layer. Images are first screened using OCR text density and simple layout heuristics. Images with insufficient textual content are classified as non-documents and never reach the LLM. Only valid documents are sent to downstream extraction models."

That sentence alone shows you're thinking like someone building a production Document AI system.

My recommendation for this assignment
Use OpenCV + Tesseract/EasyOCR to filter obvious non-documents (free).
Use a small vision LLM (GPT-4.1 Nano or Gemini Flash) only to classify ambiguous document types.
Use the LLM again only for structured JSON extraction from actual documents.

This gives you a strong baseline and a clear cost optimisation story, which is exactly what the challenge is asking for.

okay now for each and every part create a proper diagram for me like from starting to ending and how we should done this 

This assignment is basically asking you to build a production-grade Document AI pipeline. If I were submitting this to Optera, this is the architecture I would build.

Overall Pipeline
                     INPUT FOLDER
                  (All Images Mixed)
                          │
                          ▼
              ┌─────────────────────────┐
              │ 1. Image Preprocessing  │
              │ • Resize               │
              │ • Orientation Fix      │
              │ • Convert RGB          │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ 2. Document Detector    │
              │ (Cheap Stage)           │
              │                         │
              │ OCR Text Density        │
              │ + OpenCV Layout         │
              └─────────────────────────┘
                    │             │
         Non-document│             │Document
                    ▼             ▼
          Reject Immediately   Continue
Stage 1 – Image Preprocessing (Free)

Every image goes through this.

Image
   │
   ├── Resize (max 1024px)
   │
   ├── Auto Rotate
   │
   ├── Remove Noise
   │
   ├── Improve Contrast
   │
   ▼
Processed Image

Libraries

OpenCV
Pillow
numpy

Cost

£0
Stage 2 – Document Detector

This is the most important stage.

                Processed Image
                       │
                       ▼
                Tesseract OCR
                       │
          ┌────────────┴────────────┐
          │                         │
      Text Found              No Text
          │                         │
          ▼                         ▼
   Probably Document         Reject

Also check

OCR Characters

↓

>100

Document

<20

Probably Object

Example

Battery

EXIDE

INVA TEC

Only 2 words

↓

Reject

Invoice

GST

Invoice

Amount

Date

CGST

...

↓

Continue

Stage 3 – Document Type Classifier

Now we know it's a document.

Need to know which type.

Document
      │
      ▼
 Small Vision LLM
(GPT-4.1 Nano)
      │
      ▼

Return

{
  "document_type":"invoice"
}

or

{
 "document_type":"mechanic_log"
}

Possible classes

invoice

receipt

meter_reading

handwritten_log

work_report

unknown

Only ONE tiny API call.

Stage 4 – Routing
                     document_type
                            │
     ┌──────────────┬──────────────┬─────────────┐
     │              │              │
Invoice        Handwritten      Meter
     │              │              │
     ▼              ▼              ▼
Invoice Parser  OCR Parser    Meter Parser
Stage 5A – Invoice Pipeline
Invoice
   │
   ▼
OCR
   │
   ▼
Extract Text
   │
   ▼
GPT Nano
   │
   ▼
Structured JSON

Output

{
  "invoice_no":"",
  "vendor":"",
  "gst":"",
  "date":"",
  "items":[],
  "total":""
}
Stage 5B – Handwritten Pipeline
Notebook
     │
     ▼
EasyOCR
     │
     ▼
Text
     │
     ▼
GPT Nano
     │
     ▼
JSON

Output

{
 "date":"",
 "mechanic":"",
 "entries":[]
}
Stage 5C – Meter Reading Pipeline
Meter
   │
   ▼
OCR
   │
   ▼
Numbers
   │
   ▼
Regex
   │
   ▼
JSON

Example

3199.76

43.24

74

32.5

Output

{
 "amount":3199.76,
 "litres":43.24,
 "price_per_litre":74,
 "urea":32.5
}

Notice

No GPT required.

Just OCR + Regex.

Almost free.

Stage 6 – Validation

Every JSON should be checked.

JSON
   │
   ▼
Pydantic

Example

Total

must be float

Date

must be valid

Invoice number

string

If invalid

↓

Retry once

Stage 7 – Output
output/

invoice1.json

invoice2.json

meter1.json

rejected1.json

Example rejected

{
  "status":"rejected",
  "reason":"non_document"
}
Cost Logger

Every API call should be logged.

Image

↓

Model

↓

Input Tokens

↓

Output Tokens

↓

Cost

↓

Time

CSV

image,model,input,output,cost,time
invoice1,gpt4.1,520,210,£0.0041,2.1s
battery,,0,0,£0,0.2s
meter,nano,130,40,£0.0002,0.8s
Baseline Architecture
Image
   │
   ▼
GPT-4.1 Vision
   │
   ▼
JSON

Simple

Expensive

Optimized Architecture
Images
   │
   ▼
OpenCV
   │
   ▼
OCR
   │
   ▼
Document?
   │
 ┌─┴──────────────┐
 │                │
No               Yes
 │                │
 ▼                ▼
Reject      Small LLM
                  │
                  ▼
          Document Type
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
Invoice     Handwritten     Meter
      │           │           │
      ▼           ▼           ▼
 Parser      OCR+Parser    OCR+Regex
      │           │           │
      └───────────┼───────────┘
                  ▼
           Validate JSON
                  │
                  ▼
            Save Output
Suggested Folder Structure
optera-doc-ai/
│
├── input/
│
├── output/
│
├── logs/
│     cost_log.csv
│
├── models/
│
├── src/
│
│   preprocess.py
│   detector.py
│   classifier.py
│   router.py
│
│   invoice_parser.py
│   handwritten_parser.py
│   meter_parser.py
│
│   validator.py
│   cost_logger.py
│
│   schemas.py
│   utils.py
│
├── main.py
│
├── README.md
│
├── DESIGN.md
│
└── requirements.txt