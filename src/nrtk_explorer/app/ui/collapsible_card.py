from trame.widgets import quasar
from trame.widgets import html


class CollapsibleCard(quasar.QCard):
    id_count = 0

    def __init__(self, name=None, collapsed=False, **kwargs):
        super().__init__(**kwargs)

        if name is None:
            CollapsibleCard.id_count += 1
            name = f"is_card_open_{CollapsibleCard.id_count}"
            self.state.client_only(name)  # keep it local if not provided

        with self:
            with quasar.QCardSection():
                with html.Div(classes="row items-center no-wrap"):
                    self.slot_title = html.Div(classes="col")
                    with html.Div(classes="col-auto"):
                        quasar.QBtn(
                            round=True,
                            flat=True,
                            dense=True,
                            click=f"{name} = !{name}",
                            icon=(f"{name} ? 'keyboard_arrow_up' : 'keyboard_arrow_down'",),
                        )
            with quasar.QSlideTransition():
                with html.Div(v_show=(name, not collapsed)):
                    self.slot_content = quasar.QCardSection()
                    self.slot_actions = quasar.QCardActions(align="right")


class CollapsibleCardUnslotted(quasar.QCard):
    def __init__(self, name=None, collapsed=False, **kwargs):
        super().__init__(**kwargs)

        if name is None:
            CollapsibleCard.id_count += 1
            name = f"is_card_open_{CollapsibleCard.id_count}"
            self.state.client_only(name)  # keep it local if not provided

        with self:
            with quasar.QCardSection():
                with html.Div(classes="row items-center no-wrap"):
                    self.slot_title = html.Div(classes="col")
                    with html.Div(classes="col-auto"):
                        quasar.QBtn(
                            round=True,
                            flat=True,
                            dense=True,
                            click=f"{name} = !{name}",
                            icon=(f"{name} ? 'keyboard_arrow_up' : 'keyboard_arrow_down'",),
                        )
            with quasar.QSlideTransition():
                self.slot_collapse = html.Div(v_show=(name, not collapsed))
