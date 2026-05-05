/* Shared i18n — loaded by index.html and settings.html */

const TRANSLATIONS = {
  en: {
    /* ── nav / header ── */
    app_title:"Face Recognition", nav_home:"Home", nav_settings:"Settings",
    /* ── live feed ── */
    live_feed:"Live feed", loading_cameras:"Loading cameras…", cams_loaded:"cams loaded",
    cam_running:"Camera running", cam_stopped:"Camera stopped",
    /* ── visit history ── */
    visit_history:"Visit History", visit_history_sub:"Location-based attendance tracking",
    /* ── tabs ── */
    tab_analytics:"Analytics", tab_attendance:"Attendance", tab_daily:"Daily Summary",
    tab_person:"Per Person", tab_location:"Per Location",
    /* ── form labels ── */
    label_person:"Person:", label_date:"Date:", label_from:"From:", label_to:"To:",
    label_location:"Location:", label_to_range:"to", select_placeholder:"Select...",
    /* ── table columns ── */
    col_footage:"Footage", col_person:"Person", col_arrived:"Arrived", col_left:"Left",
    col_status:"Status", col_location:"Location", col_duration:"Duration",
    col_action:"Action", col_date:"Date", col_employee:"Employee", col_num:"#",
    /* ── statuses ── */
    status_active:"Active", status_ended:"Ended",
    /* ── empty / loading states ── */
    empty_person_date:"Select a person and date", empty_date:"Select a date and click Load",
    empty_person:"Select a person and click Load", empty_location:"Select a location and click Load",
    loading:"Loading...", no_data:"No data", no_data_range:"No data for this range",
    /* ── pager ── */
    prev:"Prev", next:"Next", duration_label:"Duration:", total_label:"Total:",
    /* ── analytics tiles ── */
    tile_peak_hour:"Peak Hour Today", tile_present:"Present Today",
    tile_absent:"Absent Today", tile_unknowns:"Unknowns to Resolve",
    /* ── arrivals ── */
    arrivals_by_shift:"Arrivals by Shift", morning_shift:"Morning shift",
    night_shift:"Night shift", btn_earliest:"Earliest", btn_latest:"Latest",
    /* ── analytics sections ── */
    daily_headcount:"Daily Headcount", attendance_heatmap:"Attendance Heatmap",
    longest_working:"Top 10 Longest Working", period_today:"Today",
    period_week:"This Week", period_month:"This Month", period_year:"This Year",
    people_label:"People", modal_present:"Present Today", modal_absent:"Absent Today",
    /* ── settings: general ── */
    sec_people:"People", sec_camera:"Camera",
    add_image:"Add image", add_image_sub:"Saves as 1.jpg, 2.jpg, … in faces/<person>/",
    label_existing:"Existing", label_new:"New",
    label_image:"Image", btn_upload:"Upload",
    merge_into:"Merge into:", merge_or_new:"or new:", btn_merge:"Merge", btn_clear:"Clear",
    no_people:"No people enrolled yet", no_people_sub:"Use the form above to upload face images",
    ip_cameras:"IP Cameras", ip_cameras_sub:"Group cameras under a shared RTSP base (e.g. an NVR). Add channels by number, or use the \"Standalone\" group for one-off URLs. All edits auto-save.",
    btn_add_camera:"+ Add Camera", btn_add_group:"+ Add Group",
    /* ── settings: modals ── */
    modal_rename_title:"Rename / Merge Person", label_current_name:"Current name",
    label_new_name:"New name", rename_merge_hint:"If the name already exists, photos and visits will be merged.",
    btn_cancel:"Cancel", btn_rename:"Rename",
    modal_delete_title:"Delete Person", delete_confirm:"Are you sure you want to delete",
    delete_warning:"This will remove all face images and visit history for this person. This action cannot be undone.",
    btn_delete:"Delete",
    modal_gallery_title:"Face Images", btn_select_all:"Select all",
    btn_move_selected:"Move selected", btn_delete_selected:"Delete selected",
    modal_transfer_title:"Transfer Image", label_transfer_to:"Transfer to",
    label_new_person:"Or create new person", btn_transfer:"Transfer",
    modal_add_camera_title:"Add Camera", label_name:"Name", label_rtsp_url:"RTSP URL",
    modal_take_photo:"Take Photo", btn_capture:"Capture",
    /* ── settings: ip cameras (JS-rendered) ── */
    col_name:"Name", col_channel:"Channel", col_url:"URL", col_actions:"Actions",
    btn_test:"Test", btn_delete_group:"Delete group", btn_add_row:"+ Add",
    /* ── settings: people cards (JS-rendered) ── */
    btn_rename:"Rename",
    card_image:"image", card_images:"images",
    people_enrolled:"people enrolled", people_known:"known", people_auto:"auto-captured",
    folder_hint:"Folder names are sanitized (letters/numbers/_-).",
    branch_riyadh:"Riyadh", branch_egypt:"Egypt",
  },
  ar: {
    /* ── nav / header ── */
    app_title:"نظام التعرف على الوجه", nav_home:"الرئيسية", nav_settings:"الإعدادات",
    /* ── live feed ── */
    live_feed:"البث المباشر", loading_cameras:"جارٍ تحميل الكاميرات…", cams_loaded:"كاميرا محملة",
    cam_running:"الكاميرا تعمل", cam_stopped:"الكاميرا متوقفة",
    /* ── visit history ── */
    visit_history:"سجل الزيارات", visit_history_sub:"تتبع الحضور حسب الموقع",
    /* ── tabs ── */
    tab_analytics:"التحليلات", tab_attendance:"الحضور", tab_daily:"الملخص اليومي",
    tab_person:"حسب الشخص", tab_location:"حسب الموقع",
    /* ── form labels ── */
    label_person:"الشخص:", label_date:"التاريخ:", label_from:"من:", label_to:"إلى:",
    label_location:"الموقع:", label_to_range:"إلى", select_placeholder:"اختر...",
    /* ── table columns ── */
    col_footage:"التسجيل", col_person:"الشخص", col_arrived:"وصل", col_left:"غادر",
    col_status:"الحالة", col_location:"الموقع", col_duration:"المدة",
    col_action:"النشاط", col_date:"التاريخ", col_employee:"الموظف", col_num:"#",
    /* ── statuses ── */
    status_active:"نشط", status_ended:"انتهى",
    /* ── empty / loading states ── */
    empty_person_date:"اختر شخصاً وتاريخاً", empty_date:"اختر تاريخاً",
    empty_person:"اختر شخصاً", empty_location:"اختر موقعاً",
    loading:"جارٍ التحميل...", no_data:"لا توجد بيانات", no_data_range:"لا توجد بيانات لهذه الفترة",
    /* ── pager ── */
    prev:"السابق", next:"التالي", duration_label:"المدة:", total_label:"الإجمالي:",
    /* ── analytics tiles ── */
    tile_peak_hour:"ذروة الحضور اليوم", tile_present:"الحاضرون اليوم",
    tile_absent:"الغائبون اليوم", tile_unknowns:"مجهولون للمراجعة",
    /* ── arrivals ── */
    arrivals_by_shift:"الوصول حسب الوردية", morning_shift:"الوردية الصباحية",
    night_shift:"الوردية المسائية", btn_earliest:"الأبكر", btn_latest:"الأحدث",
    /* ── analytics sections ── */
    daily_headcount:"الحضور اليومي", attendance_heatmap:"خريطة الحضور",
    longest_working:"أعلى 10 في ساعات العمل", period_today:"اليوم",
    period_week:"هذا الأسبوع", period_month:"هذا الشهر", period_year:"هذا العام",
    people_label:"أشخاص", modal_present:"الحاضرون اليوم", modal_absent:"الغائبون اليوم",
    /* ── settings: general ── */
    sec_people:"الأشخاص", sec_camera:"الكاميرا",
    add_image:"إضافة صورة", add_image_sub:"يحفظ كـ 1.jpg, 2.jpg, … في faces/<الشخص>/",
    label_existing:"موجود", label_new:"جديد",
    label_image:"الصورة", btn_upload:"رفع",
    merge_into:"دمج في:", merge_or_new:"أو جديد:", btn_merge:"دمج", btn_clear:"مسح",
    no_people:"لا يوجد أشخاص مسجلون", no_people_sub:"استخدم النموذج أعلاه لرفع صور الوجه",
    ip_cameras:"كاميرات IP", ip_cameras_sub:"تجميع الكاميرات تحت قاعدة RTSP مشتركة. جميع التعديلات تحفظ تلقائياً.",
    btn_add_camera:"+ إضافة كاميرا", btn_add_group:"+ إضافة مجموعة",
    /* ── settings: modals ── */
    modal_rename_title:"تغيير الاسم / دمج", label_current_name:"الاسم الحالي",
    label_new_name:"الاسم الجديد", rename_merge_hint:"إذا كان الاسم موجوداً، ستُدمج الصور والزيارات.",
    btn_cancel:"إلغاء", btn_rename:"تغيير الاسم",
    modal_delete_title:"حذف شخص", delete_confirm:"هل أنت متأكد من حذف",
    delete_warning:"سيتم حذف جميع صور الوجه وسجل الزيارات. لا يمكن التراجع عن هذا الإجراء.",
    btn_delete:"حذف",
    modal_gallery_title:"صور الوجه", btn_select_all:"تحديد الكل",
    btn_move_selected:"نقل المحدد", btn_delete_selected:"حذف المحدد",
    modal_transfer_title:"نقل الصورة", label_transfer_to:"نقل إلى",
    label_new_person:"أو إنشاء شخص جديد", btn_transfer:"نقل",
    modal_add_camera_title:"إضافة كاميرا", label_name:"الاسم", label_rtsp_url:"رابط RTSP",
    modal_take_photo:"التقاط صورة", btn_capture:"التقاط",
    /* ── settings: ip cameras (JS-rendered) ── */
    col_name:"الاسم", col_channel:"القناة", col_url:"الرابط", col_actions:"الإجراءات",
    btn_test:"اختبار", btn_delete_group:"حذف المجموعة", btn_add_row:"+ إضافة",
    /* ── settings: people cards (JS-rendered) ── */
    btn_rename:"تغيير الاسم",
    card_image:"صورة", card_images:"صور",
    people_enrolled:"شخص مسجل", people_known:"معروف", people_auto:"مُلتقط تلقائياً",
    folder_hint:"أسماء المجلدات تُعقَّم (حروف/أرقام/_-).",
    branch_riyadh:"الرياض", branch_egypt:"مصر",
  }
};

window._lang = localStorage.getItem("lang") || "en";

function t(key) {
  return (TRANSLATIONS[window._lang] || TRANSLATIONS.en)[key] || key;
}

function _applyLang(lang, extras) {
  window._lang = lang;
  localStorage.setItem("lang", lang);
  const html = document.documentElement;
  html.lang = lang;
  html.dir = lang === "ar" ? "rtl" : "ltr";
  document.body.style.fontFamily = lang === "ar" ? "'Cairo', sans-serif" : "";
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  const btn = document.getElementById("langToggleBtn");
  if (btn) btn.textContent = lang === "ar" ? "EN" : "AR";
  if (typeof extras === "function") extras();
}
