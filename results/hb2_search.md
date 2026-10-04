# Data-gap search: public datasets pairing panic attacks and cardiac events with wearable signals

**Claim being supported (outline §1, §2, Appendix D):** we found no public dataset that pairs confirmed panic attacks and confirmed cardiac events in the same participants, recorded with the same wearable sensors.

## What kind of search this is
- **Type: a rapid, structured web search.** It is **not** a formal systematic review: there was no PRISMA protocol, no second screener, and no exhaustive database export. The paper must describe it in these terms: "In a dated rapid search (2026-10-01) we found no…". It must not say "systematic search".
- **Date run:** 2026-10-01.
- **Method:** general web-search queries targeting dataset repositories and the literature, followed by opening every candidate that looked relevant. The earlier version of this file was a template listing databases that had not actually been queried; it is replaced by this record.

## Queries run (2026-10-01)
1. `public dataset panic attack physiological signals ECG EDA wearable recorded during panic attacks`
2. `PhysioNet database panic disorder OR "panic attack" ECG recordings`
3. `dataset chest pain emergency department panic attack versus acute coronary syndrome wearable ECG physiological open data`
4. `Zenodo OR figshare OR "IEEE DataPort" panic attack dataset heart rate wearable`
5. `"panic" "arrhythmia" OR "supraventricular tachycardia" misdiagnosed panic disorder wearable monitoring dataset`
6. `panic attack prediction wearable device heart rate next day machine learning study` (run for the related-work check)

## Inclusion criteria (all four must hold)
(a) the same participants; (b) clinically confirmed panic-attack episodes; (c) clinically confirmed cardiac events; (d) co-registered wearable physiological signals (ECG, PPG, EDA, respiration or temperature).

## Candidates found by the queries above

| Dataset / source | Panic? | Cardiac events? | Wearable signals | Verdict |
|---|---|---|---|---|
| Digital Phenotyping Dataset for Impending Panic Symptoms (Korea University; Zenodo 10.5281/zenodo.12194136; *Sci Data* 2024) | Panic diaries and questionnaires (self-report) | No | Fitbit heart rate, steps, sleep | Excluded: no cardiac events; panic is self-reported |
| Oh et al., *J Med Internet Res* 2025;27:e69045 (panic prediction from wearable ECG) | Panic disorder patients | No | Wearable ECG / HRV | Excluded: no cardiac events; public availability of the data not established |
| Tsai et al., *JMIR Med Inform* 2022;10(2):e33063 | Weekly self-reported attacks | No | Smartwatch HR, sleep, activity | Excluded: no cardiac events; study data, not a public dataset |
| Zephyr BioPatch panic-disorder recordings (systematic review, PMC10046237) | 19 panic episodes | No | Zephyr BioPatch (ECG, respiration) | Excluded: no cardiac events; not a released dataset |
| PhysioNet anxiety biosignal recordings (ECG, skin conductance, respiration) | Anxiety disorder (not confirmed attacks) | No | ECG, EDA, respiration | Excluded: no confirmed panic attacks; no cardiac events |
| Shaoxing–Ningbo 12-lead ECG database (PhysioNet/CinC 2021) | No panic labels* | Arrhythmia diagnoses | 12-lead (not wearable) | Excluded |
| 12-lead ECG dataset for ACS prediction (*Sci Data* 2026, 19,955 ECGs) | No | ACS | 12-lead (not wearable) | Excluded |
| MIT-BIH Arrhythmia, PTB-XL | No | Arrhythmia / clinical ECG labels | Holter / 12-lead | Excluded |
| WESAD | Laboratory stress (TSST) proxy | No | Chest and wrist | Excluded (the proxy dataset used in this paper) |

\*One search summary claimed this database was "used for panic attack prediction." It contains no panic labels; that claim was not supported and is ignored.

**Known proxy datasets from earlier drafts (not re-queried in this search):** PPG-DaLiA, Stress-Predict, SWELL-KW, AffectiveROAD, K-EmoCon, MIMIC-IV waveforms, OhioT1DM, EPHNOGRAM. Each is stress-, emotion- or activity-only or cardiac-only, and none meets (b) and (c) together.

## Result
**No dataset met all four criteria as of 2026-10-01.** Panic-only wearable datasets (self-reported panic, consumer heart rate) and cardiac-only datasets (12-lead or Holter arrhythmia and ACS) exist, but none pairs the two in the same participants.

## Related clinical evidence found (motivation, not data)
- Lessmeier et al., "Unrecognized paroxysmal supraventricular tachycardia: potential for misdiagnosis as panic disorder," *Arch Intern Med* 1997;157(5):537–543:
  - PSVT was unrecognised after initial evaluation in 59 of 107 patients (55%);
  - 67% of PSVT patients met DSM-IV panic disorder criteria;
  - median delay to diagnosis was 3.3 years;
  - event monitors detected PSVT in 47% of those monitored, vs 9% for Holter.
  - This is a strong clinical anchor for the paper's motivation; it has been added to outline §1.

## Rerun protocol
Before posting, rerun the queries above (and optionally Google Dataset Search and PhysioNet's own search) and update the date. If the paper is reposted more than 6 months after the date above, restate the claim as "as of [date]".

## Sources
- https://zenodo.org/records/12194136
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12526660/ (Oh et al. 2025)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8889475/ (Tsai et al. 2022)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10046237/ (Zephyr BioPatch review)
- https://www.nature.com/articles/s41597-026-07278-0 (ACS 12-lead dataset)
- https://www.physionet.org/content/mitdb/1.0.0/
- https://jamanetwork.com/journals/jamainternalmedicine/article-abstract/623013 (Lessmeier et al. 1997)
