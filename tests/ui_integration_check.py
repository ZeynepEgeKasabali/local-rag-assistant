"""Manual integration check: uses real Foundry models and the Streamlit session."""

from streamlit.testing.v1 import AppTest


app = AppTest.from_file("../app.py", default_timeout=180)
app.run()
assert len(app.exception) == 0, app.exception

app.chat_input[0].set_value("Asistan görüşme saati ne zaman?").run()
assert len(app.exception) == 0, app.exception
assert len(app.chat_message) == 2, f"İlk turda 2 balon bekleniyordu: {len(app.chat_message)}"
assert "çarşamba" in app.chat_message[1].markdown[0].value.casefold()

app.chat_input[0].set_value("Mini projede grafik arayüz gerekiyor mu?").run()
assert len(app.exception) == 0, app.exception
assert len(app.chat_message) == 4, f"İki turda 4 balon bekleniyordu: {len(app.chat_message)}"
assert "hayır" in app.chat_message[3].markdown[0].value.casefold()

print("UI_INTEGRATION_OK messages=4 history_preserved=true real_models=true")
