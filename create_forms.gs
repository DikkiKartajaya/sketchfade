/**
 * Drawing Practice Study (SketchFade) · Google Forms builder
 * Study Instruments v0.4 (docs/SketchFade_Study_Instruments_v0.4.docx)
 *
 * Creates two Google Forms in your Google Drive, each linked to its own response sheet:
 *
 *   1. Screening   I1, filled in by volunteers before enrollment
 *   2. Session     I2 consent, I4 (drawing A after the baseline), I5, I6, I4 review
 *                  (drawings A-D) and I7, filled in by the participant during the session,
 *                  with pause pages where the device goes back to the researcher (I3 v0.4)
 *
 * Not included, on purpose: the "Researcher use only" box of I1, the signature table
 * of I2, the researcher scoring of I6 (done afterwards from the answers), and the
 * facilitator instruments I3, I8 and I9.
 *
 * How to use
 *   1. Open https://script.google.com and create a new project.
 *   2. Replace everything in Code.gs with this file and save.
 *   3. Fill in CONFIG below (the [bracketed] items of I2).
 *   4. Choose the function createAll in the toolbar and press Run. Allow the requested
 *      access (Google asks to manage your forms, spreadsheets and Drive files).
 *   5. Open the Execution log: it prints, for each form, the link for participants,
 *      the edit link and the response sheet.
 *   Later: run exportJson to save every response of both forms as a JSON file in Drive
 *   (for plots or other analysis). Responses are also in the linked sheets
 *   (File > Download > CSV).
 *
 * Running createAll again makes new copies; it never changes forms made earlier.
 */

const CONFIG = {
  contact: "[email]",                              // researchers' contact
  supervisor: "[name, email]",                     // supervising instructor
  compensation: "[state amount or “none”]",
  retention: "[period]",                           // how long data are kept
  location: "[secure location]",                   // where data are kept
  interviewRecording: "[audio-recorded / noted]",  // I2 consent statement 4
  codePattern: "^[A-Za-z0-9_-]{1,32}$",           // same rule as the app's participant ID
};

const TITLE = "Drawing Practice Study";            // neutral name (blinding)
const YES_NO = ["Yes", "No"];
const NO_YES = ["No", "Yes"];

function createAll() {
  const screening = buildScreeningForm();
  const session = buildSessionForm();
  PropertiesService.getScriptProperties().setProperties({
    screeningFormId: screening.getId(),
    sessionFormId: session.getId(),
  });
  [["Screening (I1)", screening], ["Session (I2, I4-I7)", session]].forEach(([name, form]) => {
    Logger.log(`${name}\n  participants: ${form.getPublishedUrl()}\n  edit: ${form.getEditUrl()}` +
               `\n  responses: ${SpreadsheetApp.openById(form.getDestinationId()).getUrl()}`);
  });
}

// --- shared helpers ------------------------------------------------------------

function newForm(subtitle, description) {
  const form = FormApp.create(`${TITLE} · ${subtitle}`);
  form.setTitle(TITLE)
      .setDescription(description)
      .setProgressBar(true)
      .setCollectEmail(false)
      .setAllowResponseEdits(false)
      .setShowLinkToRespondAgain(false);
  const sheet = SpreadsheetApp.create(`${TITLE} · ${subtitle} (responses)`);
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
  return form;
}

function choice(form, title, choices, required = true, help = "") {
  return form.addMultipleChoiceItem().setTitle(title).setHelpText(help)
             .setChoiceValues(choices).setRequired(required);
}

function participantCode(form) {
  form.addTextItem()
      .setTitle("Participant code")
      .setHelpText("The code the researcher gave you, e.g. P01.")
      .setRequired(true)
      .setValidation(FormApp.createTextValidation()
        .requireTextMatchesPattern(CONFIG.codePattern)
        .setHelpText("Letters, digits, - or _ only, as given by the researcher.")
        .build());
}

/** A page of its own: the next content must start with its own page break. */
function pause(form) {
  form.addPageBreakItem()
      .setTitle("Please stop here")
      .setHelpText("Hand the device back to the researcher. The researcher will tell you when to " +
                   "continue with the next page.");
}

// --- form 1: screening (I1) ------------------------------------------------------

function buildScreeningForm() {
  const form = newForm("Screening",
    "Thank you for your interest. We are recruiting adults who are beginners at portrait drawing. " +
    "Your answers decide whether the study suits you, and do not affect any grade.");
  form.setConfirmationMessage("Thank you. The researchers will contact you about taking part.");
  form.addTextItem()
      .setTitle("Name or contact")
      .setHelpText("Kept separately from study data.")
      .setRequired(true);
  choice(form, "1. Are you 18 years or older?", YES_NO);
  choice(form, "2. Have you had formal training in portrait drawing?", ["None", "A little", "A lot"]);
  form.addTextItem()
      .setTitle("3. How many years of drawing experience do you have?")
      .setHelpText("Years (0 if none).")
      .setRequired(true)
      .setValidation(FormApp.createTextValidation()
        .requireNumberBetween(0, 100).setHelpText("Please enter a number of years.").build());
  choice(form, "4. How often have you drawn in the last 6 months?",
         ["Never", "Rarely", "Monthly", "Weekly or more"]);
  choice(form, "5. How familiar are you with drawing on a stylus tablet?",
         ["Not at all", "Slightly", "Moderately", "Very"]);
  choice(form, "6. Have you used AI drawing or image tools?", ["Never", "A few times", "Often"]);
  choice(form, "7. Can you use a stylus tablet for about 90 minutes, with short breaks?", YES_NO);
  choice(form, "8. Can you attend one session of about 2 hours?", YES_NO);
  choice(form, "9. Did you take part in the pilot of this study?", YES_NO);
  return form;
}

// --- form 2: session (I2, I4, I5, I6, I4 review, I7) ---------------------------------

const I4_CRITERIA = ["Proportions", "Feature placement", "Structural coherence", "Line clarity"];
const I4_RUBRIC =
  "Compare your drawing with the reference. Scores 2 and 4 fall between the descriptions. " +
  "Tick one box for each criterion.\n\n" +
  "Proportions\n" +
  "  1 Low: Major distortion in head or feature sizes.\n" +
  "  3 Intermediate: Broad proportions recognizable, with several visible mismatches.\n" +
  "  5 High: Head and relative feature sizes closely match the reference.\n\n" +
  "Feature placement\n" +
  "  1 Low: Several features are misplaced or misaligned.\n" +
  "  3 Intermediate: Most features in plausible locations, with noticeable spacing errors.\n" +
  "  5 High: Feature positions, spacing and alignments closely match.\n\n" +
  "Structural coherence\n" +
  "  1 Low: Parts are missing or disconnected in a way that obscures the face.\n" +
  "  3 Intermediate: A coherent face with local structural inconsistencies.\n" +
  "  5 High: Parts form a complete, coherent structure matching the reference.\n\n" +
  "Line clarity\n" +
  "  1 Low: Unintended overlapping or uncertain marks obscure forms.\n" +
  "  3 Intermediate: Forms are readable despite some repeated or unclear marks.\n" +
  "  5 High: Intentional, readable lines communicate the relevant forms.";

/** One I4 sheet on its own page. `when` keeps the titles unique (baseline vs review). */
function drawingSelfAssessment(form, label, when, newPage = true) {
  const title = `Drawing ${label} (${when})`;
  if (newPage) {
    form.addPageBreakItem().setTitle(title)
        .setHelpText("Rate the drawing beside its reference. There are no right or wrong answers.");
  }
  form.addSectionHeaderItem().setTitle("Rubric").setHelpText(I4_RUBRIC);
  form.addGridItem()
      .setTitle(`${title}: scores`)
      .setHelpText("1 = low, 3 = intermediate, 5 = high")
      .setRows(I4_CRITERIA)
      .setColumns(["1", "2", "3", "4", "5"])
      .setRequired(true);
  form.addParagraphTextItem()
      .setTitle(`${title}: what would you improve in this drawing? (optional)`)
      .setRequired(false);
}

function buildSessionForm() {
  const form = newForm("Session",
    "Please answer each page when the researcher asks you to. At a “Please stop here” page, " +
    "hand the device back to the researcher.");
  form.setConfirmationMessage("Thank you. Please hand the device back to the researcher.");

  // page 1 · participant code + I2 information and consent
  participantCode(form);
  form.addSectionHeaderItem()
      .setTitle("Information sheet")
      .setHelpText(
        "Study title: Drawing Practice Study\n" +
        "Researchers: Berlian Angga Kusuma, Dikki Yusuf Kartajaya and Muhammad Ariful Furqon, " +
        `Japan Advanced Institute of Science and Technology. Contact: ${CONFIG.contact}. ` +
        `Supervising instructor: ${CONFIG.supervisor}.\n\n` +
        "What is this study about?\n" +
        "We are studying how a computer tool can support people who are learning to draw portraits.\n\n" +
        "What will I do?\n" +
        "You will attend one session of about 2 hours, with short breaks. You will draw portraits on a " +
        "stylus tablet, rate your own drawings, complete short questionnaires and take part in a brief " +
        "interview. Some drawing tasks are done without any help from the tool.\n\n" +
        "Your rights\n" +
        "• Taking part is voluntary and has no connection to any grade.\n" +
        "• You may stop at any time without giving a reason and without penalty.\n" +
        "• Reference portraits are synthetic or licensed images, not photographs of you.\n" +
        "• Risks are low. You may feel mild tiredness or frustration, and may take a break at any time.\n" +
        `• Compensation: ${CONFIG.compensation}.\n\n` +
        "Your data\n" +
        "• We record your drawings, pen strokes, tool use, ratings of your own drawings, questionnaire " +
        "answers and interview responses.\n" +
        "• Your data are stored under a code. This form is stored separately from that code.\n" +
        `• Data will be kept for ${CONFIG.retention} at ${CONFIG.location}, then deleted.\n` +
        "• Results may appear in course reports and publications without identifying you.");
  const consent = [
    ["I have read this information and could ask questions.", true],
    ["I understand that taking part is voluntary and that I may stop at any time.", true],
    ["I agree that my drawings and interaction records are used for this study.", true],
    [`I agree to my interview being ${CONFIG.interviewRecording} and transcribed.`, true],
    ["I agree that anonymized data may be shared with other researchers. (optional)", false],
  ];
  form.addSectionHeaderItem().setTitle("Consent").setHelpText("Please tick each statement you agree with.");
  consent.forEach(([text, required]) => {
    form.addCheckboxItem().setTitle(text).setChoiceValues(["Yes"]).setRequired(required);
  });

  // after the stylus warm-up and the baseline (I3 steps 2-3)
  pause(form);

  // I3 step 4 · I4 for the baseline drawing, labelled A
  form.addPageBreakItem().setTitle("Your first drawing")
      .setHelpText("Rate the drawing you just made (labelled A on the tablet) beside its reference. " +
                   "There are no right or wrong answers.");
  drawingSelfAssessment(form, "A", "after the baseline", false);

  // practice blocks, break, post-test (I3 steps 5-8)
  pause(form);

  // I3 step 9 · I5 then I6
  form.addPageBreakItem().setTitle("Your experience")
      .setHelpText("Think about the drawing practice with the tool in this session.");
  form.addGridItem()
      .setTitle("Experience with the practice")
      .setHelpText("1 = strongly disagree · 7 = strongly agree. One answer per statement.")
      .setRows([
        "1. I feel that the drawing is my own work.",
        "2. I felt in control of how the drawing developed.",
        "3. The guidance helped me decide what to draw next.",
        "4. I feel able to start a similar portrait without help.",
        "5. The amount of help matched what I needed.",
      ])
      .setColumns(["1", "2", "3", "4", "5", "6", "7"])
      .setRequired(true);
  form.addPageBreakItem().setTitle("Workload")
      .setHelpText("Rate the practice with the tool in this session. Mark one box on each scale " +
                   "(1 = left end, 20 = right end). Note that Performance runs from Good to Poor.");
  form.addGridItem()
      .setTitle("Workload of the practice")
      .setRows([
        "Mental Demand. How mentally demanding was the task? (1 very low – 20 very high)",
        "Physical Demand. How physically demanding was the task? (1 very low – 20 very high)",
        "Temporal Demand. How hurried or rushed was the pace of the task? (1 very low – 20 very high)",
        "Performance. How successful were you in accomplishing what you were asked to do? (1 good – 20 poor)",
        "Effort. How hard did you have to work to accomplish your level of performance? (1 very low – 20 very high)",
        "Frustration. How insecure, discouraged, irritated, stressed and annoyed were you? (1 very low – 20 very high)",
      ])
      .setColumns(Array.from({ length: 20 }, (_, i) => String(i + 1)))
      .setRequired(true);

  // retention and transfer (I3 steps 10-11)
  pause(form);

  // I3 step 12 · I4 review of drawings A-D (the tablet shows them in a random order)
  form.addPageBreakItem().setTitle("Drawing review")
      .setHelpText("Here are drawings you made in this session. Rate each one beside its reference, " +
                   "using the same sheet as before. Each page is for the drawing with that label on the tablet.");
  ["A", "B", "C", "D"].forEach((label, i) => drawingSelfAssessment(form, label, "review", i > 0));

  // I3 step 13 · I7
  form.addPageBreakItem().setTitle("Before and during this session");
  choice(form, "During the breaks, did you practice drawing?", NO_YES);
  form.addTextItem()
      .setTitle("If yes: about how many minutes in total?")
      .setRequired(false)
      .setValidation(FormApp.createTextValidation()
        .requireNumberBetween(0, 600).setHelpText("Please enter a number of minutes.").build());
  choice(form, "Before today, had you seen any portrait used in this study?", NO_YES);
  choice(form, "During the breaks, did you look at another drawing tool or tutorial?", NO_YES);
  form.addTextItem().setTitle("If yes: which tool or tutorial?").setRequired(false);
  choice(form, "Before today, did anyone tell you about the tasks?", NO_YES);
  return form;
}

// --- export ---------------------------------------------------------------------

/** Every response of both forms as one JSON file in Drive (Drawing Practice Study responses <time>.json). */
function exportJson() {
  const ids = PropertiesService.getScriptProperties().getProperties();
  const out = { exported: new Date().toISOString(), forms: {} };
  [["screening", ids.screeningFormId], ["session", ids.sessionFormId]].forEach(([name, id]) => {
    if (!id) return;
    const form = FormApp.openById(id);
    out.forms[name] = form.getResponses().map(r => {
      const answers = {};
      r.getItemResponses().forEach(ir => {
        const item = ir.getItem(), value = ir.getResponse();
        if (item.getType() === FormApp.ItemType.GRID) {   // {row: answer}
          const rows = item.asGridItem().getRows(), byRow = {};
          rows.forEach((row, k) => { byRow[row] = value[k]; });
          answers[item.getTitle()] = byRow;
        } else {
          answers[item.getTitle()] = value;
        }
      });
      return { submitted: r.getTimestamp().toISOString(), answers };
    });
  });
  const stamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyyMMdd_HHmmss");
  const file = DriveApp.createFile(`${TITLE} responses ${stamp}.json`,
                                   JSON.stringify(out, null, 1), MimeType.PLAIN_TEXT);
  Logger.log(`Saved ${file.getName()}: ${file.getUrl()}`);
}
