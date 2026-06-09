package com.example.moodtender

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.moodtender.data.HealthDataRequest
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 🚀 앱이 켜지자마자 바로 통신 테스트 시작!
        sendTestDataToServer()
    }

    private fun sendTestDataToServer() {
        // 1. 가짜 테스트 데이터 만들기
        val testData = HealthDataRequest(
            recordDate = "2026-06-09",
            stepCount = 7777, // 테스트용 걸음수
            sleepMinutes = 400,
            screenTimeMinutes = 120,
            appUsageJson = mapOf("kakao" to 60, "youtube" to 30),
            depressionScore = 0
        )

        // 2. Retrofit 밸브를 통해 서버로 전송
        RetrofitClient.instance.sendHealthData(testData).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    // 성공했을 때 안드로이드 스튜디오 하단 로그캣(Logcat)에 초록색으로 출력됨
                    Log.d("통신테스트", "✅ 성공! 서버 응답: ${response.body()}")
                } else {
                    Log.e("통신테스트", "❌ 실패: 서버에서 거절함 (에러코드 ${response.code()})")
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                // 인터넷이 안 되거나 서버가 꺼져있을 때 출력됨
                Log.e("통신테스트", "💥 서버 접속 아예 실패: ${t.message}")
            }
        })
    }
}