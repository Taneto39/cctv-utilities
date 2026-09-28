import base64
import json
import os
import urllib.request

from cloudevents.http.event import CloudEvent
import functions_framework

from google.api_core import exceptions
from google.cloud import billing_v1
from google.cloud import logging

billing_client = billing_v1.CloudBillingClient()


def get_project_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id is not None:
        return project_id

    url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")
    project_id = urllib.request.urlopen(req).read().decode()

    if project_id is None:
        raise ValueError("project-id metadata not found.")

    return project_id


@functions_framework.cloud_event
def stop_billing(cloud_event: CloudEvent) -> None:
    SIMULATE_DEACTIVATION = False

    PROJECT_ID = get_project_id()
    PROJECT_NAME = f"projects/{PROJECT_ID}"

    event_data = base64.b64decode(
        cloud_event.data["message"]["data"]
    ).decode("utf-8")

    event_dict = json.loads(event_data)
    cost_amount = event_dict["costAmount"]
    budget_amount = event_dict["budgetAmount"]
    print(f"Cost: {cost_amount} Budget: {budget_amount}")

    if cost_amount <= budget_amount:
        print("No action required. Current cost is within budget.")
        return

    print(f"Disabling billing for project '{PROJECT_NAME}'...")

    is_billing_enabled = _is_billing_enabled(PROJECT_NAME)

    if is_billing_enabled:
        _disable_billing_for_project(PROJECT_NAME, SIMULATE_DEACTIVATION)
    else:
        print("Billing is already disabled.")


def _is_billing_enabled(project_name: str) -> bool:
    try:
        print(f"Getting billing info for project '{project_name}'...")
        response = billing_client.get_project_billing_info(name=project_name)
        return response.billing_enabled
    except Exception as e:
        print(f'Error getting billing info: {e}')
        print("Unable to determine if billing is enabled, assuming enabled.")
        return True


def _disable_billing_for_project(project_name: str, simulate_deactivation: bool) -> None:
    logging_client = logging.Client()
    logger = logging_client.logger(name="disable-billing")

    if simulate_deactivation:
        entry_text = "Billing disabled. (Simulated)"
        print(entry_text)
        logger.log_text(entry_text, severity="CRITICAL")
        return

    try:
        project_billing_info = billing_v1.ProjectBillingInfo(billing_account_name="")
        response = billing_client.update_project_billing_info(
            name=project_name, project_billing_info=project_billing_info
        )
        entry_text = f"Billing disabled: {response}"
        print(entry_text)
        logger.log_text(entry_text, severity="CRITICAL")
    except exceptions.PermissionDenied:
        print("Failed to disable billing, check permissions.")