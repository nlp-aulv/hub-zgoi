外部数据源的总结：
1. 搜索引擎：
DuckDuckGo: 通过 duckduckgo_search 库用于一般网页搜索。
搜狗: 通过 sogou_search 库用于一般网页搜索。
Bing: 直接通过 requests 访问 bing.com 进行网页搜索，并通过 方案-天池三轮车/ 中的 BingSearchEngine 类。
百度搜索: 在 stock_bd.py 中通过 finance.pae.baidu.com 获取股票和板块信息，以及通过 方案-天池三轮车/ 中的 baidusearch 库。
中国新闻网: 抓取新闻文章。
2. 财经数据提供商：
AkShare: 这是一个主要的金融数据来源，被广泛用于所有“方案”目录中，以获取：
香港（HK）和A股（上海和深圳）公司的财务报表（资产负债表、利润表、现金流量表）。
股票介绍和基本信息。
财务指标和摘要。
期货数据（futures_hist_em）。
外汇数据（forex_spot_em、forex_hist_em）。
宏观经济数据（通过 macro_akshare.py 中动态调用的 akshare 方法）。
同花顺 (10jqka.com.cn): 直接抓取股东信息，存在于 方案-好想成为人类/ 和 方案-天池三轮车/ 的 get_shareholder_info.py 中。
东方财富网: 通过 AkShare 作为期货和外汇数据来源。
上海证券交易所: 在 report_sh.py 中直接抓取公司报告和公告。
深圳证券交易所: 在 report_sz.py 中直接抓取公司报告和公告。
香港交易所 (HKEX): 在 report_hk.py 中直接抓取公司报告和公告，并在 index_hs.py 中抓取恒生指数数据。
3. 政府数据来源：
中国人民银行: 从 pbc.gov.cn 抓取统计数据，存在于 data_rmyh.py 中。
中国国家统计局: 从 data.stats.gov.cn 抓取各种月度和季度宏观经济统计数据，存在于 data_gjtjj.py 中。
国务院政策文件库: 从 sousuo.www.gov.cn 抓取政府政策文件，存在于 zhengce_gwy.py 中。
中国人民政府网: 从 sousuoht.www.gov.cn 抓取政府文件，存在于 zhengce_rmzf.py 中。
4. 其他：
雪球: 在 stock_xq.py 中引用了股票相关数据。
finnews.cubenlp.com: 一个用于新闻搜索的API，在 方案-队伍名字不能为空/ 的 news.py 中使用。