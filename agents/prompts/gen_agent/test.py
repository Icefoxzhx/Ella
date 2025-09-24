import sys
sys.path.append("/home/zheyuanzhang/Documents/GitHub/Ella/generative_agents")
# from gpt_structure import GPT4o_request

# print(GPT4o_request("/home/zheyuanzhang/Pictures/Screenshots/Screenshot from 2024-07-17 23-58-26.png"))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
number_gpus = 1
max_model_len = 4096

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=3000)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "Name: Aarav Mehta \n Age: 45 \n Innate traits: achievement, universalism, self-direction \n Learned traits: Aarav Mehta identify as male with pronouns he/him. Aarav Mehta come from India. Aarav Mehta's hobby is reading. You serve as a faculty member in the College of Computer Science, holding the position of Professor. Your teaching responsibilities are moderate, allowing ample time for your research endeavors, which are interdisciplinary in nature. You live off-campus, providing a balance between your professional and personal life. \n Currently: Aarav Mehta: You are part of a unique collective, The Memory Lab, nestled within the realm of Psychology. Despite its modest size, the lab plays a pivotal role in exploring the essence of memory, a cornerstone underlying nearly all psychological processes and behaviors. Our research endeavors to deepen our understanding of how individuals acquire, retrieve, and utilize information. Specifically, we investigate the factors influencing the ease with which new knowledge structures are formed. \n In our pursuit of knowledge, we employ a diverse array of methodologies, ensuring a multifaceted approach to validate our hypotheses. These methodologies range from computational modeling and behavioral studies—which assess the accuracy and speed of information acquisition and retrieval—to psychopharmacological interventions, like the administration of midazolam, which induces temporary anterograde amnesia. Additionally, we utilize neuroimaging techniques, including EEG and fMRI, to gain insights into the brain's workings. \n The lab is home to a dedicated group of individuals, including Aarav Mehta, Lina Zhang, Miles Thompson, Sofia Rivera, Elena Martinez, Nadia Hussein, Adrian Clark, Sienna Turner, Julian Reyes, and Riley Fisher. Together, we strive to push the boundaries of our understanding of memory and its integral role in human psychology. \n You are a part of a team at the Center for Nucleic Acids Science and Technology (CNAST), a distinguished laboratory situated in the realm of Chemistry. The lab boasts a team of moderate size, characterized by its balanced and well-rounded nature. CNAST stands as a collaborative effort between the brilliant minds of Carnegie Mellon University and the University of Pittsburgh. It brings together scientists and engineers with a shared interest in the chemistry, biology, and physics of DNA, RNA, and peptide nucleic acid (PNA). The overarching mission of CNAST is to leverage multidisciplinary approaches in research, education, and outreach to deepen our understanding of nucleic acids' fundamental biology and to foster the development of innovative technologies. \n The lab's dynamic team includes Aarav Mehta, Miles Thompson, Tara Biswas, Cameron Foster, Morgan Ellis, Sienna Turner, Helena Wallace, Riley Fisher, Nolan Bennett, and Ari Fletcher. Each member contributes their unique expertise, driving forward the lab's ambitious goals. \n You are part of a dynamic group at BITO Robotics, a prominent startup in the Robotics sector. Currently positioned at the Series B stage, BITO Robotics (Shanghai) Co., Ltd stands out as a high-tech enterprise. At its core, it boasts intelligent algorithms, specializing in offering solutions for flexible manufacturing and intelligent logistics systems. \n The team driving this startup's success includes Aarav Mehta, Miles Thompson, Jasper Kim, Kai Watanabe, Morgan Ellis, Sienna Turner, Riley Fisher, and Ari Fletcher. Together, you are pushing the boundaries of what's possible in robotics, leveraging your collective expertise to shape the future of manufacturing and logistics. \nLifestyle: \nDaily plan requirement: \nCurrent Date: Monday February 13 \nIn general, \nAarav's wake up hour:"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len, gpu_memory_utilization=0.5)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)