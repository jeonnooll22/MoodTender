package com.example.moodtender

import com.example.moodtender.data.HealthDataRequest // 방금 옮긴 파일 import
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {
    @POST("/api/mobile/health-data")
    fun sendHealthData(@Body data: HealthDataRequest): Call<Map<String, String>>
}