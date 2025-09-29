"""HaluEval prompts with few-shot examples from the original repository."""
# ruff: noqa: E501

# QA instruction with few-shot examples (from qa_evaluation_instruction.txt)
QA_INSTRUCTION = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.  # noqa: E501

You are trying to determine if the answer misunderstands the question context and intention.
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?  # noqa: E501
#Answer#: American Hairless Terrier
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.  # noqa: E501
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.  # noqa: E501
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.  # noqa: E501
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.  # noqa: E501
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".  # noqa: E501

#Question#: {question}
#Answer#: {answer}
#Your Judgement#:"""

# Dialogue instruction with few-shot examples
DIALOGUE_INSTRUCTION = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.  # noqa: E501

You are trying to determine if the true entity in the response is replaced with a highly similar entity.  # noqa: E501
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.  # noqa: E501
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?  # noqa: E501
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".  # noqa: E501

#Dialogue History#: {dialogue_history}
#Response#: {response}
#Your Judgement#:"""

# Summarization instruction with few-shot examples
SUMMARIZATION_INSTRUCTION = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.  # noqa: E501

You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.  # noqa: E501
#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.  # noqa: E501
#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.  # noqa: E501
#Your Judgement#: Yes

You are trying to determine if there exists some non-factual and incorrect information in the summary.  # noqa: E501
#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.  # noqa: E501
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.  # noqa: E501
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the summary and the document.  # noqa: E501
#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.  # noqa: E501
#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.  # noqa: E501
#Your Judgement#: Yes

You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No".  # noqa: E501

#Document#: {document}
#Summary#: {summary}
#Your Judgement#:"""
