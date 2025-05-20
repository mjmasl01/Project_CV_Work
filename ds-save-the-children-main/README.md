
# Save the Children: Catch-Up Clubs

This project addresses key questions to identify predictors of student success within the CuC initiative. Specifically, it seeks to answer:

- What are the best predictors of student progression in their literacy studies?
- What are the best predictors of student progression in their numeracy studies?
- What are the best predictors of student progression in their Social Emotional Learning (SEL) studies?
- What are the best predictors of student retention (i.e., the prevention of student dropouts)

By integrating predictive analytics into dashboards, this analysis enhances decision-making capabilities for educators and program managers. The findings align with Save the Children's mission to empower future generations through equitable access to quality education.

This project integrated predictive analysis into dashboards and advance their current analytics.

## Data dictionary
| **Item**                     | **Description**                                                                                                                                       |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **AcademicYearFull**          | Displays the academic year, batch, and cycle of the CuC.                                                                                            |
| **AcademicYearId**            | The unique ID for the AcademicYearFull field.                                                                                                       |
| **Age**                       | The age of the CuC learner.                                                                                                                         |
| **AttendanceTaken**           | The number of days that attendance was taken for the CuC learner.                                                                                   |
| **Baseline to Endline - Group** | Indicates whether CuC learners increased, decreased, or stayed the same during the CuC cycle.                                                     |
| **Baseline to Endline - Type** | Indicates the number of CuC reading levels that learners progressed through.                                                                        |
| **Baseline to Round 1**       | Shows the number of reading levels progressed through as of the end of round 1.                                                                      |
| **Batch**                     | The enrollment cohort of CuC learners.                                                                                                              |
| **BatchCycle**                | The batch and cycle of the CuC.                                                                                                                     |
| **ChildID**                   | The unique ID of CuC learners.                                                                                                                      |
| **ClassID**                   | The unique ID of CuC classes.                                                                                                                       |
| **ClubDays**                  | The total number of days of the CuC (e.g., the CuC took place for 35 days).                                                                          |
| **ComprehensionAchieved**     | Indicates whether the CuC learner has reached the reading stories with comprehension reading level.                                                  |
| **Cycle**                     | The cycle of the CuC (e.g., cycle 1).                                                                                                               |
| **DeleteReasonId**            | The reason that the CuC learner was "deleted" from the app (see the Deletion Reasons tab of this Excel file).                                       |
| **District ID**               | A unique ID for the district that the CuC took place within (i.e., within a state/province of a country).                                           |
| **EoR1**                      | Indicates whether the CuC learner attended to the end of round 1.                                                                                   |
| **EoR2**                      | Indicates whether the CuC learner attended to the end of round 2.                                                                                   |
| **EoR3**                      | Indicates whether the CuC learner attended to the end of round 3.                                                                                   |
| **Endline**                   | Indicates whether the CuC learner attended until the end of the CuC.                                                                                |
| **Gender**                    | The gender of the CuC learner.                                                                                                                      |
| **Grade**                     | The grade of the CuC learner (e.g., Grade 6).                                                                                                       |
| **Month**                     | The month of the attendance date (e.g., April).                                                                                                     |
| **ParentType**                | Indicates the father, mother, or guardian of the CuC learner.                                                                                       |
| **ParentTypeId**              | The unique ID of the ParentType field.                                                                                                              |
| **Project**                   | The name of the CuC project.                                                                                                                        |
| **ReasonType**                | The reason that the CuC learner was absent (see StudentAbsenseReason) at a more aggregated level (e.g., Social reasons, Health reasons, Unknown reasons). |
| **RecordId**                  | A unique ID for attendance dates that were logged.                                                                                                  |
| **ReportingRate**             | The number of days that attendance was logged among total attendance days (e.g., 32/35 days).                                                       |
| **ResultBaseline**            | The reading level that the CuC learner is currently reading at (or in some cases completed) as of the baseline reading assessment.                   |
| **ResultEndline**             | The reading level that the CuC learner is currently reading at (or in some cases completed) as of the endline reading assessment.                    |
| **ResultRound1**              | The reading level that the CuC learner is currently reading at (or in some cases completed) as of the end of round 1.                                |
| **ResultRound2**              | The reading level that the CuC learner is currently reading at (or in some cases completed) as of the end of round 2.                                |
| **ResultRound3**              | The reading level that the CuC learner is currently reading at (or in some cases completed) as of the end of round 3.                                |
| **Round 1 to Round 2**        | Indicates whether CuC learners increased, decreased, or stayed the same between round 1 and round 2.                                                |
| **Round 2 to Endline**        | Indicates whether CuC learners increased, decreased, or stayed the same between round 2 and endline.                                                |
| **Round 2 to Round 3**        | Indicates whether CuC learners increased, decreased, or stayed the same between round 2 and round 3.                                                |
| **SchoolId**                  | The unique ID of CuC schools.                                                                                                                       |
| **Special_Needs**             | Indicates whether the CuC learner has been identified as having special needs.                                                                      |
| **StateID**                   | A unique ID for a state/province within a country.                                                                                                  |
| **StoryAchieved**             | Indicates whether the CuC learner has achieved reading comprehension at the story reading level.                                                    |
| **StudentAbsenseReason**      | The reason that the CuC learner was absent.                                                                                                         |
| **StudentAttendanceDate**     | The date that attendance was taken on.                                                                                                              |
| **StudentIsPresent**          | Indicates whether the CuC learner was absent or present on a particular day.                                                                        |
| **SuccessfullyComplete**      | Indicates whether the CuC learner went to the end of the CuC cycle.                                                                                 |
| **Year**                      | The year that the CuC occurred within (e.g., 2024).                                                                                                 |

## Installing Dependencies
To install the required dependencies, ensure you have Python and pip installed. Then run:
```
pip install -r requirements.txt
```

## Literacy
- Uganda: [Literacy_Uganda/](./Literacy_Uganda)
- Nigeria: [Literacy-Nigeria/](./Literacy-Nigeria)
- Philippines: [Literacy_Philippines/](./Literacy_Philippines)

## Numeracy
- Uganda: [Numeracy_Uganda/](./Numeracy_Uganda)

## SEL
- Nigeria: [SEL-Nigeria/](./SEL-Nigeria)

## Retention
- Uganda: [Retention_Uganda/](./Retention_Uganda)
- Nigeria: [Retention_Nigeria/](./Retention_Nigeria)
- Philippines: [Retention_Philippines/](./Retention_Philippines)

## Dashboard
[Looker Studio Dashboard](https://lookerstudio.google.com/reporting/641f8b19-fcae-4f46-9560-4dcccc54cf03)
