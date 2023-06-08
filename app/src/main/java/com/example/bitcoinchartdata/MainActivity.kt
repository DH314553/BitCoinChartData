package com.example.bitcoinchartdata

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import okhttp3.*
import okhttp3.Request.Builder
import org.datavec.api.split.NumberedFileInputSplit
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.IOException
import java.util.*

class MainActivity : AppCompatActivity() {
    private var chart: LineChart = LineChart(applicationContext)
    private val featureBaseDir =
        "/Users/daisaku/AndroidStudioProjects/BitCoinChartData2/app/src/debug/assets/feature"
    private val targetsBaseDir =
        "/Users/daisaku/AndroidStudioProjects/BitCoinChartData2/app/src/debug/assets"
    private var production: BitcoinPriceProduction? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        chart = findViewById(R.id.chart_view)
        production = BitcoinPriceProduction()
        production!!.net.init()
        val client = OkHttpClient()
        val request: Request = Builder()
            .url("https://api.bitflyer.com/v1/getexecutions?product_code=BTC_JPY")
            .build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                val body = Objects.requireNonNull(response.body).toString()
                var jsonArray: JSONArray? = null
                try {
                    jsonArray = JSONArray(body)
                } catch (e: JSONException) {
                    e.printStackTrace()
                }
                val entries: MutableList<Entry> = ArrayList()
                for (i in 0 until Objects.requireNonNull<JSONArray?>(jsonArray).length()) {
                    var jsonObject: JSONObject
                    try {
                        jsonObject = jsonArray!!.getJSONObject(i)
                        val price = jsonObject.getDouble("price").toFloat()
                        val timestamp = jsonObject.getString("exec_date")
                        val time = "00000000000000000"
                        entries.add(
                            Entry(
                                time.replace("/[^\\p{Nd}]/u".toRegex(), timestamp).toFloat(), price
                            )
                        )
                    } catch (e: JSONException) {
                        e.printStackTrace()
                    }
                }
                runOnUiThread {
                    val dataSet = LineDataSet(entries, "BTC/JPY")
                    val data = LineData(dataSet)
                    chart.data = data
                    chart.invalidate()
                }
            }
        })
    }

    private fun readCsv() = try {
        val eval = RegressionEvaluation()
        production!!.trainFeatures.initialize(
            NumberedFileInputSplit(
                "$featureBaseDir/%d.csv",
                1,
                1936
            )
        )
        production!!.trainTargets.initialize(
            NumberedFileInputSplit(
                "$targetsBaseDir/bitcoin.csv",
                1,
                1936
            )
        )
        production!!.testFeatures.initialize(
            NumberedFileInputSplit(
                "$featureBaseDir/%d.csv",
                1937,
                2089
            )
        )
        production!!.testTargets.initialize(
            NumberedFileInputSplit(
                "$targetsBaseDir/bitcoin.csv",
                1937,
                2089
            )
        )
        production!!.net.init()
        production!!.net.fit(production!!.train, 15)
        production!!.test.reset()
        production!!.train.reset()
        while (production!!.test.hasNext()) {
            val next = production!!.test.next()
            val features = next.features as NDArray
            var pred: NDArray = Nd4j.zeros(1, 2) as NDArray
            var i: Int
            i = 0
            while (i < 50) {
                pred = production!!.net.rnnTimeStep(
                    features.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + 1)
                    )
                ) as NDArray
                i++
            }
            val correct = production!!.train.next()
            val cFeatures = correct.features as NDArray
            i = 0
            while (i < 10) {
                eval.evalTimeSeries(
                    pred,
                    cFeatures.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval(i, i + 1)
                    )
                )
                pred = production!!.net.rnnTimeStep(pred) as NDArray
                i++
            }
            production!!.net.rnnClearPreviousState()
        }
    } catch (e: IOException) {
        throw RuntimeException(e)
    } catch (e: InterruptedException) {
        throw RuntimeException(e)
    }
}