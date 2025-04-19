from gql import gql

START_INTERVIEW = gql(
    """
mutation StartInterview($userId: String!, $role: String!) {
  startInterview(userId: $userId, role: $role) {
    id
  }
}
"""
)

END_INTERVIEW = gql(
    """
mutation EndInterview($interviewId: String!) {
  endInterview(interviewId: $interviewId)
}
"""
)

SAVE_INTERVIEW_DATA = gql(
    """
mutation SaveInterviewData($data: SaveInterviewDataInput!) {
  saveInterviewData(interviewData: $data)
}
"""
)

SAVE_INTERVIEW_ANALYSIS = gql(
    """
mutation SaveInterviewAnalysis($interviewId: String!, $analysis: String!) {
  saveInterviewAnalysis(interviewId: $interviewId, analysis: $analysis) {
    id
    interviewId
    analysis
  }
}
"""
)
