# IBM Z Datathon 2026: form answers (copy-paste)

## 1. What is the problem you are trying to solve?
During floods, disaster control rooms receive hundreds of SOS calls, drone sightings and helpline relays every hour. They are usually handled first-come-first-served or with a paper checklist. As a result, the most critical cases (severe injuries, rising water, collapsing buildings, people who need insulin or dialysis) wait behind stable ones, and rescue boats and drones go to the wrong places first. At the same time, victims' names, phone numbers and medical details pass through spreadsheets and chat groups with no protection. We are solving real-time prioritisation and dispatch for flood response, with victim data kept secure.

## 2. How are you planning to solve the problem? (Brief overview)
PAVAN SEWA Command is a real-time triage and dispatch engine that grew out of our PAVAN SEWA disaster-response drone.
1. **Ingest:** one REST API takes SOS calls, scout-drone detections (head-count, water depth and rise rate, collapse risk) and helpline relays.
2. **Triage AI:** an explainable multinomial model classifies each incident as CRITICAL, URGENT or STABLE. It returns a 0–100 priority and the top factors behind it, in microseconds.
3. **Dispatch optimizer:** proposes which medical drone (6 kg kit) or rescue boat (capacity 10) should go where, using priority first and ETA second.
4. **Human-in-the-loop dashboard:** a live map and priority queue, where an operator approves each dispatch. Low-confidence predictions are flagged for review.
5. **Privacy vault:** names and phone numbers are AES-encrypted at rest. Operators see only pseudonyms and masked numbers. Only the rescue-lead role can decrypt, and every access is audited.

On held-out data, critical recall is 85%, against 68% for a rule checklist. In a simulated 6-hour flood surge with the same fleet, 65% of critical cases are reached before harm, against 17% under first-come-first-served, and the median critical wait falls from 271 to 47 minutes. (The data is synthetic and modelled on the Cauvery/Kollidam flood zone at Tiruchirappalli; the pipeline is built to retrain on real disaster-authority logs.)

## 3. How does your solution leverage IBM Z / LinuxONE technologies?
- **Runs natively on IBM LinuxONE (s390x).** The full stack (API, AI inference, encryption, dispatch, storage, dashboard) runs on one LinuxONE VM. A setup script and a multi-arch container are included. Measured on our LinuxONE VM (s390x, RHEL 9): 17.6 µs median / 30.3 µs p99 per triage, 54,172 incidents per second on one core, and a 150-incident dispatch plan in 2.8 ms.
- **AI next to the data.** Inference runs in-process on the same system that stores the incidents. There is no external AI API, sensitive data never leaves the platform, and latency is in microseconds.
- **Telum on-chip AI path.** The model is exported to ONNX using only core operators (MatMul, Add, Softmax), so the IBM Z Deep Learning Compiler (zDLC, `--maccel=NNPA`) can compile it for the Integrated Accelerator for AI on z16 and later. A pure-Python engine is the portable fallback.
- **Hardware-backed security.** AES field encryption goes through OpenSSL, which uses the CPACF crypto instructions on IBM Z, on top of pervasive encryption. The design is ready for Secure Execution / Hyper Protect.
- **Availability and scale.** A disaster command centre must stay up in the middle of a disaster. Mainframe-class reliability, and the ability to consolidate every district control room onto one secure data store (Db2 / PostgreSQL on LinuxONE), fit this workload.

## 4. What value does your solution deliver, and who will benefit from it?
- **Flood victims:** the most critical people are reached first (3.9× more critical cases reached in time in our simulation), and their personal data stays protected.
- **State and district disaster control rooms (SDMA/DDMA):** one live, ranked, explained picture across every channel, instead of a call queue.
- **NDRF/SDRF rescue teams and drone operators:** clear assignments with ETAs and fewer wasted trips.
- **Hospitals and medical teams:** early warning of medical-need cases.
- **Authorities and auditors:** a full audit trail of every AI recommendation, human approval and data access.

The same engine extends to cyclones, landslides and earthquakes.

## 8. Links to open-source resources used
- Python: https://www.python.org
- Flask: https://github.com/pallets/flask
- NumPy: https://github.com/numpy/numpy
- pandas: https://github.com/pandas-dev/pandas
- scikit-learn: https://github.com/scikit-learn/scikit-learn
- cryptography (Fernet / AES): https://github.com/pyca/cryptography
- SQLite: https://www.sqlite.org
- ONNX: https://github.com/onnx/onnx
- ONNX Runtime (parity check): https://github.com/microsoft/onnxruntime
- IBM Z Deep Learning Compiler (deployment path): https://github.com/IBM/zDLC
- Playwright (demo recording): https://github.com/microsoft/playwright
- pytest: https://github.com/pytest-dev/pytest
