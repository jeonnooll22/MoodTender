package com.example.moodtender.data

data class HealthDataRequest(
    @SerializedName("record_date")
    val recordDate: String,

    @SerializedName("step_count")
    val stepCount: Int,

    @SerializedName("sleep_minutes")
    val sleepMinutes: Int,

    @SerializedName("screen_time_minutes")
    val screenTimeMinutes: Int,

    @SerializedName("app_usage_json")
    val appUsageJson: Map<String, Int>,

    @SerializedName("depression_score")
    val depressionScore: Int? = null
)