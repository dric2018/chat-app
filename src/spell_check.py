from pprint import pprint 

from utils import get_corrected_context, get_entity_context


if __name__=="__main__":
    user_query = input("Ask anything: ")
    corrections = get_corrected_context(user_query)
    print(corrections)

    chat_history = get_entity_context(user_query)
    pprint(chat_history)