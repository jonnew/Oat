# NOTE: Function argument OatCommon_LIBS is a LIST and therefore needs to be
# quoted or only the first element will be passed

add_oat_test (Helpers       "${OatCommon_LIBS}")
add_oat_test (Node          "${OatCommon_LIBS}")
add_oat_test (Sink          "${OatCommon_LIBS}")
add_oat_test (Source        "${OatCommon_LIBS}")
add_oat_test (concurrency   "${OatCommon_LIBS}")
