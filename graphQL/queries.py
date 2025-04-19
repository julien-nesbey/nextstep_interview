from gql import gql

GET_INTERVIEW_DATA = gql(
    """
query GetInterviewData($interviewId: String!) {
  getInterviewData(interviewId: $interviewId) {
    id
    questions {
      id
      text
    }
    responses {
      id
      questionId
      answer
    }
  }
}
"""
)
