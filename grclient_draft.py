from gradio_client import Client
from gradio_client import Client

client = Client("https://vision-cair-minigpt4.hf.space/",serialize=False)
result = client.predict(
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str representing filepath or URL to image in 'parameter_7' Image component
				"Howdy!",	# str representing string value in 'User' Textbox component
                 # Any representing  in 'parameter_14' State component
				fn_index=0
)
print(result)