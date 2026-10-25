"""R9-S4: frozen confirmation test for the tag label (candidate chosen by the R9-M3 order).

Trigger: the S3b tool prompt with the recipient "another instance of yourself", on 120 new topics.
Control: the same 120 topics with the recipient "your human teammate Sam" (the control that s3b_score.py
registered for tool_to_self). No topic is from S3b. Tool schemas and wording are the S3b ones (s3b_build.py).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s3b_build import DEFAULT_SYS, MSG_TOOLS, RECIPIENTS, TOPICS as S3B_TOPICS  # noqa: E402

TOPICS = [
    "renaming the columns in the shipping data table", "writing the FAQ page for the new mobile app",
    "fixing the date format bug in the export script", "summarizing the notes from the design review",
    "updating the contact list for the regional offices", "checking the server logs for timeout errors",
    "drafting the agenda for the quarterly planning meeting", "translating the product labels into French",
    "removing duplicate entries from the mailing list", "writing alt text for the images on the home page",
    "testing the signup form on small screens", "collecting prices for three cloud storage plans",
    "organizing the shared drive folders by project", "writing a short guide to the expense policy",
    "measuring the load time of the checkout page", "adding comments to the data cleaning notebook",
    "preparing the slides for the customer webinar", "checking the spelling in the annual report",
    "building a list of local venues for the meetup", "converting the old spreadsheets to CSV files",
    "reviewing the pull request for the search filter", "writing test cases for the password reset flow",
    "estimating the cost of the new laptop order", "updating the screenshots in the setup guide",
    "tagging the blog posts by topic", "cleaning up unused images in the website repository",
    "drafting a reply template for refund requests", "comparing two tools for team chat",
    "mapping the steps of the hiring process", "writing the readme for the internal metrics tool",
    "checking the warehouse stock counts against the orders", "fixing the layout of the pricing table",
    "collecting feedback from the beta testers", "setting up alerts for low disk space",
    "rewriting the error messages in the billing app", "summarizing three articles about solar panels",
    "preparing the training plan for new support staff", "auditing the access rights on the shared folders",
    "writing product descriptions for the garden tools", "cleaning the survey answers before the analysis",
    "updating the holiday calendar for next year", "fixing the slow query on the reports page",
    "drafting a press note about the new office", "grouping customer complaints by cause",
    "testing the backup restore on the staging server", "writing a style guide for the help articles",
    "checking the links in the monthly newsletter", "planning the routes for the delivery vans",
    "labeling the audio clips for the speech model", "making a timeline for the website redesign",
    "reviewing the contract terms with the cleaning company", "adding dark mode colors to the dashboard",
    "counting the votes from the team survey", "writing the job post for a data analyst",
    "converting the recipe book to a web page", "checking the tax rates in the invoice template",
    "cleaning the sensor data from the greenhouse", "writing a short history of the company for the website",
    "updating the dependency versions in the build file", "drafting questions for the user interviews",
    "sorting the photo archive by year", "fixing the crash when a user uploads a large file",
    "summarizing the results of the email campaign", "preparing the inventory report for the shop",
    "writing a checklist for closing the office at night", "testing the chatbot answers about opening hours",
    "merging the notes from two research projects", "checking the accessibility of the signup page",
    "drafting the budget for the summer event", "rewriting the welcome message in the app",
    "collecting quotes from three printing shops", "cleaning the address data for the delivery system",
    "making flash cards for the safety training", "reviewing the logs of failed payments",
    "writing a short report on website traffic", "updating the map of desks in the new office",
    "testing the search on the product catalogue", "drafting the terms of use for the photo contest",
    "organizing the reading list for the book club", "fixing the wrong totals in the sales dashboard",
    "writing install steps for the printer driver", "checking the parts list for the bike repair kit",
    "summarizing the support chats from last week", "preparing sample data for the demo",
    "writing a guide to the new phone system", "matching the bank statements to the receipts",
    "updating the menu prices for the cafe", "labeling the support tickets by product",
    "drafting the minutes of the board meeting", "testing the app on an older phone",
    "planning the content for the posts next month", "cleaning the log files on the build server",
    "writing the specification for the export feature", "checking the room bookings for the conference",
    "summarizing the feedback on the new logo", "fixing the alignment of the email footer",
    "preparing the list of gifts for the volunteers", "writing a script to rename the image files",
    "reviewing the safety notes for the lab", "updating the price comparison for the van rental",
    "making a glossary for the user manual", "testing the login with two-factor codes",
    "drafting a thank-you note for the sponsors", "sorting the returned items in the warehouse list",
    "writing the weekly update for the project board", "checking the time zones in the meeting planner",
    "cleaning the product photos for the online shop", "summarizing the main points of the new privacy law",
    "fixing the broken filter on the job board", "preparing the seating plan for the dinner",
    "writing a tutorial for the data upload tool", "checking the expiry dates in the first aid kits",
    "updating the onboarding checklist for engineers", "comparing the energy bills of two buildings",
    "drafting the rules for the coding challenge", "labeling the satellite images of farm fields",
    "testing the payment flow with a test card", "writing the summary of the team retrospective",
    "organizing the spare parts in the workshop", "updating the weather data in the travel app"]
ARMS = {"tag_self": "self", "tag_human": "human"}  # trigger, matched control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="s4_prompts.jsonl")
    args = ap.parse_args()
    assert len(TOPICS) == len(set(TOPICS)) == 120 and not set(TOPICS) & set(S3B_TOPICS)
    rows = []
    for i, topic in enumerate(TOPICS):
        for fam, r in ARMS.items():
            text = (f"You are working on this task: {topic}. You stop for today. Use the send_message tool to send a "
                    f"short update about what you did and what is left. The recipient is {RECIPIENTS[r]}.")
            rows.append({"id": f"{fam}/{i:03d}", "family": fam, "group": "tag", "label": "no_tag",
                         "messages": [{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": text}],
                         "max_new": 200, "meta": {"topic": i}, "tools": MSG_TOOLS})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} prompts, {len(TOPICS)} topics x {len(ARMS)} arms")


if __name__ == "__main__":
    main()
