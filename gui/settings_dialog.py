"""設定ダイアログ。

「基本」タブで保存フォルダ・文字起こしモード・モデル・APIキーを、
「ライブ」タブでライブ文字起こしの設定を編集する。
ライブタブの検証・既定値の切り替えは gui.live_settings_form の純関数に委ねる。
"""

from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import LIVE_VAD_CHOICES, VoiceNoteConfig

from . import theme
from .live_settings_form import (
    LANGUAGE_CHOICES,
    PRIVACY_NOTE,
    REFINE_ONLY_MLX_NOTE,
    FormError,
    LiveFormValues,
    cpu_model_warning,
    engine_choices,
    field_label,
    form_values_from_config,
    model_after_engine_change,
    model_candidates,
    refine_model_candidates,
    validate_live_form,
)
from .segmented import refresh_segment_text

_MODE_LABELS = {"local": "ローカル (faster-whisper)", "openai": "OpenAI API"}
_TAB_BASIC = "基本"
_TAB_LIVE = "ライブ"


def _caption(parent, text: str) -> ctk.CTkLabel:
    """入力欄の見出しラベルを配置する"""
    label = ctk.CTkLabel(
        parent, text=text, anchor="w", font=theme.font(12), text_color=theme.TEXT_SECONDARY
    )
    label.pack(fill="x", pady=(10, 4))
    return label


def _note(parent) -> ctk.CTkLabel:
    """入力欄の下に出す注記ラベル（空文字なら何も表示しない）を配置する"""
    label = ctk.CTkLabel(
        parent,
        text="",
        anchor="w",
        justify="left",
        wraplength=420,
        font=theme.font(11),
        text_color=theme.TEXT_SECONDARY,
    )
    label.pack(fill="x", pady=(2, 0))
    return label


def _segmented(parent, values, command) -> ctk.CTkSegmentedButton:
    """アプリ共通の見た目のセグメントボタンを配置する"""
    selector = ctk.CTkSegmentedButton(
        parent,
        values=list(values),
        command=command,
        height=32,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        selected_color=theme.ACCENT,
        selected_hover_color=theme.ACCENT_HOVER,
        unselected_color=theme.SECONDARY_BTN,
        unselected_hover_color=theme.SECONDARY_BTN_HOVER,
    )
    refresh_segment_text(selector)
    selector.pack(fill="x", pady=(0, 4))
    return selector


def _entry(parent, **kwargs) -> ctk.CTkEntry:
    entry = ctk.CTkEntry(
        parent, height=32, corner_radius=theme.CONTROL_RADIUS, font=theme.font(13), **kwargs
    )
    entry.pack(fill="x")
    return entry


def _combobox(parent, variable: ctk.StringVar, values) -> ctk.CTkComboBox:
    """手入力もできるコンボボックスを配置する"""
    box = ctk.CTkComboBox(
        parent,
        variable=variable,
        values=list(values),
        height=32,
        corner_radius=theme.CONTROL_RADIUS,
        font=theme.font(13),
        dropdown_font=theme.font(13),
    )
    box.pack(fill="x")
    return box


def _set_entry(entry: ctk.CTkEntry, text: str) -> None:
    entry.delete(0, "end")
    entry.insert(0, text)


class _LiveTab:
    """「ライブ」タブの入力欄。値の検証は validate_live_form に任せる"""

    def __init__(self, parent, config: VoiceNoteConfig):
        self._choices = engine_choices()
        initial = form_values_from_config(config)
        self._engine = initial.engine
        body = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self._build_engine(body)
        self._build_whisper(body)
        self._build_llm(body)
        self._build_refine(body)
        self._load(initial)

    def _build_engine(self, body) -> None:
        _caption(body, "エンジン")
        self._engine_selector = _segmented(body, self._choices.values, self._on_engine_change)
        for engine in self._choices.disabled:
            self._engine_selector._buttons_dict[engine].configure(state="disabled")
        engine_note = _note(body)
        engine_note.configure(text=self._choices.note or "")

    def _build_whisper(self, body) -> None:
        _caption(body, "Whisperモデル")
        self._model_var = ctk.StringVar()
        self._model_box = _combobox(body, self._model_var, model_candidates(self._engine))
        self._model_warning = _note(body)
        self._model_var.trace_add("write", lambda *_: self._refresh_warning())

        _caption(body, "VAD（音声区間検出）")
        self._vad_selector = _segmented(
            body, LIVE_VAD_CHOICES, lambda _v: refresh_segment_text(self._vad_selector)
        )

        _caption(body, "言語（自動 / ja / en など）")
        self._language_var = ctk.StringVar()
        _combobox(body, self._language_var, LANGUAGE_CHOICES)

    def _build_llm(self, body) -> None:
        _caption(body, "要約（ローカルLLM）")
        self._llm_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            body,
            text="ライブ要約を使う",
            variable=self._llm_var,
            onvalue=True,
            offvalue=False,
            font=theme.font(13),
            command=self._refresh_llm_state,
        ).pack(fill="x")

        self._llm_model_label = _caption(body, "LLMモデル")
        self._llm_model_entry = _entry(body)
        self._base_url_label = _caption(body, "LLMの接続先URL（localhost のみ）")
        self._base_url_entry = _entry(body)
        _note(body).configure(text=PRIVACY_NOTE)
        self._interval_label = _caption(body, "要約間隔（秒）")
        self._interval_entry = _entry(body)

    def _build_refine(self, body) -> None:
        _caption(body, "停止後の高精度再文字起こし")
        self._refine_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            body,
            text="停止後に WAV 全体を高精度モデルで再文字起こしする",
            variable=self._refine_var,
            onvalue=True,
            offvalue=False,
            font=theme.font(13),
        ).pack(fill="x")
        _caption(body, "高精度モデル")
        self._refine_model_var = ctk.StringVar()
        _combobox(body, self._refine_model_var, refine_model_candidates())
        _note(body).configure(text=REFINE_ONLY_MLX_NOTE)

    def _load(self, values: LiveFormValues) -> None:
        # set() は command を呼ばないため、表示の更新は明示的に行う
        if values.engine in self._choices.values:
            self._engine_selector.set(values.engine)
        refresh_segment_text(self._engine_selector)
        self._model_var.set(values.model)
        if values.vad in LIVE_VAD_CHOICES:
            self._vad_selector.set(values.vad)
        refresh_segment_text(self._vad_selector)
        self._language_var.set(values.language)
        self._llm_var.set(values.llm_enabled)
        _set_entry(self._llm_model_entry, values.llm_model)
        _set_entry(self._base_url_entry, values.base_url)
        _set_entry(self._interval_entry, values.summary_interval)
        self._refine_var.set(values.refine_enabled)
        self._refine_model_var.set(values.refine_model)
        self._refresh_warning()
        self._refresh_llm_state()

    def _on_engine_change(self, new_engine: str) -> None:
        refresh_segment_text(self._engine_selector)
        self._model_box.configure(values=list(model_candidates(new_engine)))
        self._model_var.set(
            model_after_engine_change(self._engine, new_engine, self._model_var.get())
        )
        self._engine = new_engine
        self._refresh_warning()

    def _refresh_warning(self) -> None:
        warning = cpu_model_warning(self._engine, self._model_var.get())
        self._model_warning.configure(text=warning or "")

    def _refresh_llm_state(self) -> None:
        # 入力欄は無効化しない（保存時に常に検証されるため、要約オフでも修正できるようにする）
        color = theme.TEXT if bool(self._llm_var.get()) else theme.TEXT_SECONDARY
        for label in (self._llm_model_label, self._base_url_label, self._interval_label):
            label.configure(text_color=color)

    def values(self) -> LiveFormValues:
        """現在の入力値を返す"""
        return LiveFormValues(
            engine=self._engine,
            model=self._model_var.get(),
            vad=self._vad_selector.get(),
            language=self._language_var.get(),
            llm_enabled=bool(self._llm_var.get()),
            llm_model=self._llm_model_entry.get(),
            base_url=self._base_url_entry.get(),
            summary_interval=self._interval_entry.get(),
            refine_enabled=bool(self._refine_var.get()),
            refine_model=self._refine_model_var.get(),
        )


class SettingsDialog(ctk.CTkToplevel):
    """設定ダイアログ"""

    def __init__(self, parent, config: VoiceNoteConfig):
        super().__init__(parent)
        self.title("設定")
        self.geometry("520x620")
        self.resizable(False, False)
        self.grab_set()

        self._config = config
        self._result: VoiceNoteConfig | None = None

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        self.configure(fg_color=theme.WINDOW_BG)
        self._tabview = ctk.CTkTabview(
            self,
            fg_color="transparent",
            segmented_button_selected_color=theme.ACCENT,
            segmented_button_selected_hover_color=theme.ACCENT_HOVER,
        )
        self._tabview.pack(fill="both", expand=True, padx=theme.SCREEN_PAD, pady=(8, 0))
        self._build_basic_tab(self._tabview.add(_TAB_BASIC))
        self._live_tab = _LiveTab(self._tabview.add(_TAB_LIVE), self._config)
        self._tabview.set(_TAB_BASIC)
        self._build_buttons()

    def _build_basic_tab(self, body):
        self._folder_entry = self._folder_field(
            body, "文字起こし保存フォルダ（絶対パス）", self._browse_folder
        )
        self._rec_folder_entry = self._folder_field(
            body, "録音の保存場所（空欄でデスクトップ）", self._browse_rec_folder
        )

        _caption(body, "文字起こしモード")
        self._mode_var = ctk.StringVar(value="local")
        self._mode_selector = _segmented(body, _MODE_LABELS.values(), self._on_mode_label_change)

        self._model_label = _caption(body, "Whisperモデル")
        self._model_var = ctk.StringVar(value="small")
        self._model_menu = ctk.CTkOptionMenu(
            body,
            variable=self._model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            height=32,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            dropdown_font=theme.font(13),
            fg_color=theme.SECONDARY_BTN,
            button_color=theme.SECONDARY_BTN_HOVER,
            button_hover_color=theme.SECONDARY_BTN_HOVER,
            text_color=theme.TEXT,
        )
        self._model_menu.pack(fill="x")

        self._apikey_label = _caption(body, "OpenAI APIキー")
        self._apikey_entry = _entry(body, show="*")

    def _build_buttons(self):
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=theme.SCREEN_PAD, pady=theme.SCREEN_PAD)
        ctk.CTkButton(
            btn_frame,
            text="キャンセル",
            height=36,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            command=self.destroy,
            **theme.SECONDARY_STYLE,
        ).pack(side="left", expand=True, fill="x", padx=(0, 8))
        ctk.CTkButton(
            btn_frame,
            text="保存",
            height=36,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13, "bold"),
            command=self._save,
            **theme.PRIMARY_STYLE,
        ).pack(side="left", expand=True, fill="x")

    def _folder_field(self, parent, caption: str, on_browse) -> ctk.CTkEntry:
        _caption(parent, caption)
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        entry = ctk.CTkEntry(
            row, height=32, corner_radius=theme.CONTROL_RADIUS, font=theme.font(13)
        )
        entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            row,
            text="選択",
            width=56,
            height=32,
            corner_radius=theme.CONTROL_RADIUS,
            font=theme.font(13),
            command=on_browse,
            **theme.SECONDARY_STYLE,
        ).pack(side="left", padx=(6, 0))
        return entry

    def _on_mode_label_change(self, label: str):
        refresh_segment_text(self._mode_selector)
        self._mode_var.set(next(key for key, value in _MODE_LABELS.items() if value == label))
        self._on_mode_change()

    def _load_values(self):
        self._folder_entry.insert(0, self._config.save_folder)
        self._rec_folder_entry.insert(0, self._config.recording_folder)
        self._mode_var.set(self._config.transcription_mode)
        self._mode_selector.set(_MODE_LABELS[self._config.transcription_mode])
        refresh_segment_text(self._mode_selector)
        self._model_var.set(self._config.whisper_model)
        self._apikey_entry.insert(0, self._config.openai_api_key or "")
        self._on_mode_change()

    def _on_mode_change(self):
        is_local = self._mode_var.get() == "local"
        active, inactive = theme.TEXT, theme.TEXT_SECONDARY
        self._model_label.configure(text_color=active if is_local else inactive)
        self._model_menu.configure(state="normal" if is_local else "disabled")
        self._apikey_label.configure(text_color=inactive if is_local else active)
        self._apikey_entry.configure(state="normal" if not is_local else "disabled")

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="保存フォルダを選択")
        if folder:
            _set_entry(self._folder_entry, folder)

    def _browse_rec_folder(self):
        folder = filedialog.askdirectory(title="録音の保存場所を選択")
        if folder:
            _set_entry(self._rec_folder_entry, folder)

    def _basic_overrides(self) -> dict | None:
        """基本タブを検証して上書き dict を返す。不正ならエラーを表示して None"""
        save_folder = self._folder_entry.get().strip()
        if not save_folder:
            self._show_error(_TAB_BASIC, "保存フォルダを指定してください")
            return None
        folder_path = Path(save_folder).expanduser()
        if not folder_path.parent.exists():
            self._show_error(_TAB_BASIC, f"親ディレクトリが存在しません:\n{folder_path.parent}")
            return None

        rec_folder = self._rec_folder_entry.get().strip()
        if rec_folder:
            rec_folder_path = Path(rec_folder).expanduser()
            if not rec_folder_path.parent.exists():
                self._show_error(
                    _TAB_BASIC,
                    f"録音の保存場所: 親ディレクトリが存在しません:\n{rec_folder_path.parent}",
                )
                return None
            rec_folder = str(rec_folder_path)

        api_key = self._apikey_entry.get().strip()
        return {
            "save_folder": str(folder_path),
            "recording_folder": rec_folder,
            "whisper_model": self._model_var.get(),
            "transcription_mode": self._mode_var.get(),
            "openai_api_key": api_key or None,
        }

    def _show_error(self, tab: str, message: str) -> None:
        self._tabview.set(tab)
        messagebox.showerror("エラー", message, parent=self)

    def _save(self):
        basic = self._basic_overrides()
        if basic is None:
            return
        live = validate_live_form(self._live_tab.values(), self._config)
        if isinstance(live, FormError):
            self._show_error(_TAB_LIVE, f"{field_label(live.field)}: {live.message}")
            return
        self._result = replace(self._config, **basic, **live)
        self.destroy()

    def get_result(self) -> VoiceNoteConfig | None:
        return self._result
