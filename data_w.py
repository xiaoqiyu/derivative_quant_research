from WindPy import w
w.start()
w_wss_data = w.vba_wss("300866.SZ",
                       "tot_oper_rev,oper_rev,net_profit_is,tot_assets,inventories,acct_rcv,arturndays,apturndays,invturn,invturndays,grossprofitmargin,expensetosales,cogstosales,netprofitmargin,ev,mkt_cap_ard,pe_ttm,ps_ttm,ev2_to_ebitda,val_evtoebitda2",
                       "unit=1;rptDate=20211231;rptType=1;tradeDate=20220703")
print(w_wss_data)