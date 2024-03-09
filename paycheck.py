import pandas as pd

Salary=150000
Age_Now=25
Inflation_Rate=0.02
Monthly_Income=7500
Monthly_Expenses=5000
Time_Frame_Desired=48
Mortgage_Down_Payment_Amt=187500
Mortgage_Down_Payment_Savings_Current=30000
Mortgage_Down_Payment_Savings_Monthly=1000
Rate_of_Return=0.04
t401K_Tiers=2
t401K_Tier1_Match_Pct=1
t401K_Tier1_Up_To_Pct=0.03
t401K_Tier2_Match_Pct=0.5
t401K_Tier2_Up_To_Pct=0.03
t401K_Deferral_Max_Pct=0.13
t401K_Deferral_Min_Pct=0.06
t401K_Contribution_Current_Pct=0.04
t401K_Company_Match_Max_Pct=0.045
Years_Until_Retirement=40
Retirement_Savings_Needed=3719982
Monthly_Retirement_Savings_Needed=2938
Current_Retirement_Savings=50000
Other_Retirement_Savings_Monthly=0
Debt_Credit_Card_Balance=15000
Debt_Credit_Card_Interest_APR=0.16
Debt_Credit_Card_Payment_Monthly=400
Debt_Student_Loan_Balance=50000
Debt_Student_Loan_Interest_APR=0.06
Debt_Student_Loan_Payment_Monthly=500
Debt_Student_Loan_Payment_Additional=200
Emergency_Fund_Months_Needed=12
Emergency_Fund_Savings_Current=0
Monthly_IRA_Contribution_Max=500
Monthly_IRA_Contribution_Current=0

#helper function to initialize data frames
def create_df(rows,cols,init_vals):
    temp_df=pd.DataFrame(0.0,index=rows,columns=cols)
    for i,val in enumerate(init_vals):
        temp_df.iloc[i].loc[0]=val
    return temp_df

#classes for various goals, each goal should have an initializer, which can be the default
#constructor, or not (see 401k), and an updater to be called by each time step forward
#Each initalizer calls create_df(rows,cols,init_vals) to create its own data frame
class Emergency:
    def __init__(self,ind,spendable,cash,expenses,current_savings):
        savings_needed=3*expenses
        spend_used=min(spendable,savings_needed-current_savings)
        balance=current_savings+spend_used
        cash_left=min(spendable,spendable-spend_used)
        cash[0]=cash_left
        self.df1=create_df(["Savings Needed","Current Savings","Monthly Spendable Used","Savings Balance","Cash Left for Distribution"],ind,[savings_needed,current_savings,spend_used,balance,cash_left])
        
    def fund2_init(self,ind,spendable,cash,expenses,months_needed):
        savings_needed=expenses*months_needed
        current_savings=self.df1.loc["Savings Balance"].loc[0]
        spend_used=min(spendable,savings_needed-current_savings)
        balance=current_savings+spend_used
        cash_left=min(spendable,spendable-spend_used)
        cash[0]=cash_left
        self.df2=create_df(["Savings Needed","Current Savings","Monthly Spendable Used","Savings Balance","Cash Left for Distribution"],ind,[savings_needed,current_savings,spend_used,balance,cash_left])
                
    def fund1_update(self,i,spendable):
        self.df1.loc["Savings Needed"].loc[i]=self.df1.loc["Savings Needed"].loc[i-1]
        self.df1.loc["Current Savings"].loc[i]=self.df1.loc["Savings Balance"].loc[i-1]
        self.df1.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.df1.loc["Savings Needed"].loc[i]-self.df1.loc["Current Savings"].loc[i])
        self.df1.loc["Savings Balance"].loc[i]=self.df1.loc["Current Savings"].loc[i]+self.df1.loc["Monthly Spendable Used"].loc[i]
        self.df1.loc["Cash Left for Distribution"].loc[i]=min(spendable,spendable-self.df1.loc["Monthly Spendable Used"].loc[i])
        return self.df1.loc["Cash Left for Distribution"].loc[i]
    
    def fund2_update(self,i,spendable):
        self.df2.loc["Savings Needed"].loc[i]=self.df2.loc["Savings Needed"].loc[i-1]
        self.df2.loc["Current Savings"].loc[i]=self.df2.loc["Savings Balance"].loc[i-1]+self.df1.loc["Monthly Spendable Used"].loc[i]
        self.df2.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.df2.loc["Savings Needed"].loc[i]-self.df2.loc["Current Savings"].loc[i])
        self.df2.loc["Savings Balance"].loc[i]=self.df2.loc["Current Savings"].loc[i]+self.df2.loc["Monthly Spendable Used"].loc[i]
        self.df2.loc["Cash Left for Distribution"].loc[i]=min(spendable,spendable-self.df2.loc["Monthly Spendable Used"].loc[i])
        return self.df2.loc["Cash Left for Distribution"].loc[i]

class c_401k:
    def __init__(self,ind,spendable,cash,cur_pct,min_pct,salary):
        current_def=cur_pct
        def_needed=max(0,min_pct-cur_pct)
        spend_needed=salary*def_needed/12
        spend_used=min(spendable,spend_needed)
        cash_left=min(spendable,spendable-spend_used)
        cash[0]=cash_left
        self.dmatch=create_df(["Current Deferral","Deferral Needed","Spendable Needed","Monthly Spendable Used","Cash Left for Distribution"],ind,[cur_pct,def_needed,spend_needed,spend_used,cash_left])
        
    def deferral_init(self,ind,spendable,cash,salary,max_pct):
        current_def=self.dmatch.loc["Current Deferral"].loc[0]+self.dmatch.loc["Monthly Spendable Used"].loc[0]/(salary/12)
        current_cont=(salary/12)*current_def
        max_def=salary*max_pct/12
        spend_used=min(spendable,(max_def-current_cont))
        cash_left=min(spendable,spendable-spend_used)
        cash[0]=cash_left
        self.ddeferral=create_df(["Current Deferral","Current Monthly Contribution","Maximum Deferral","Monthly Spendable Used","Cash Left for Distribution"],ind,[current_def,current_cont,max_def,spend_used,cash_left])
        
    def stats_init(self,ind,salary,tier1_match,tier1_upto,tier2_match,tier2_upto,match_max):
        current_def=self.ddeferral.loc["Current Deferral"].loc[0]
        pct=min(match_max,min(current_def, tier1_upto)*tier1_match+(current_def>tier1_upto)*min(tier2_upto, current_def-tier1_upto)*tier2_match)
        self.stats=create_df(["Match %","Match $"],ind,[pct,salary*pct/12])
        
    def match_update(self,i,spendable,min_pct,salary):
        self.dmatch.loc["Current Deferral"].loc[i]=(self.ddeferral.loc["Current Monthly Contribution"].loc[i-1]+self.ddeferral.loc["Monthly Spendable Used"].loc[i-1])/(salary/12)
        self.dmatch.loc["Deferral Needed"].loc[i]=max(0,min_pct-self.dmatch.loc["Current Deferral"].loc[i])
        self.dmatch.loc["Spendable Needed"].loc[i]=salary*self.dmatch.loc["Deferral Needed"].loc[i]/12
        self.dmatch.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.dmatch.loc["Spendable Needed"].loc[i])
        self.dmatch.loc["Cash Left for Distribution"].loc[i]=min(spendable,spendable-self.dmatch.loc["Monthly Spendable Used"].loc[i])
        return self.dmatch.loc["Cash Left for Distribution"].loc[i]
    
    def deferral_update(self,i,spendable,salary,max_pct):
        self.ddeferral.loc["Current Deferral"].loc[i]=(self.dmatch.loc["Current Deferral"].loc[i]*(salary/12)+self.dmatch.loc["Monthly Spendable Used"].loc[i])/(salary/12)
        self.ddeferral.loc["Current Monthly Contribution"].loc[i]=(salary/12)*self.ddeferral.loc["Current Deferral"].loc[i]
        self.ddeferral.loc["Maximum Deferral"].loc[i]=salary*max_pct/12
        self.ddeferral.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.ddeferral.loc["Maximum Deferral"].loc[i]-self.ddeferral.loc["Current Monthly Contribution"].loc[i])
        self.ddeferral.loc["Cash Left for Distribution"].loc[i]=min(spendable,spendable-self.ddeferral.loc["Monthly Spendable Used"].loc[i])
        return self.ddeferral.loc["Cash Left for Distribution"].loc[i]
    
    def stats_update(self,i,salary,tier1_match,tier1_upto,tier2_match,tier2_upto,match_max):
        current_def=self.ddeferral.loc["Current Deferral"].loc[i]
        self.stats.loc["Match %"].loc[i]=min(match_max,min(current_def, tier1_upto)*tier1_match+(current_def>tier1_upto)*min(tier2_upto, current_def-tier1_upto)*tier2_match)
        self.stats.loc["Match $"].loc[i]=self.stats.loc["Match %"].loc[i]*salary/12
#Debt has both credit card debt and student loans
class Debt:
    def __init__(self,ind,spendable,cash,amount,interest,payment):
        expense=amount*interest/12
        spend_used=max(0,min(spendable,amount+expense-payment))
        post=max(0,amount+expense-(payment+spend_used))
        cash_left=spendable-spend_used
        cash[0]=cash_left
        self.df=create_df(["Current Outstanding","Interest Expense","Payment","Monthly Spendable Used","Post Outstanding","Cash Left for Distribution"],ind,[amount,expense,payment,spend_used,post,cash_left])
        
    def update(self,i,spendable,interest,payment):
        self.df.loc["Current Outstanding"].loc[i]=self.df.loc["Post Outstanding"].loc[i-1]
        self.df.loc["Interest Expense"].loc[i]=interest*self.df.loc["Current Outstanding"].loc[i]/12
        self.df.loc["Payment"].loc[i]=payment
        self.df.loc["Monthly Spendable Used"].loc[i]=max(0,min(spendable,self.df.loc["Current Outstanding"].loc[i]+self.df.loc["Interest Expense"].loc[i]-payment))
        self.df.loc["Post Outstanding"].loc[i]=max(0,self.df.loc["Current Outstanding"].loc[i]+self.df.loc["Interest Expense"].loc[i]-(payment+self.df.loc["Monthly Spendable Used"].loc[i]))
        self.df.loc["Cash Left for Distribution"].loc[i]=spendable-self.df.loc["Monthly Spendable Used"].loc[i]
        
        debt_paid=0
        if(self.df.loc["Post Outstanding"].loc[i]==0 and self.df.loc["Post Outstanding"].loc[i-1]!=0):
            debt_paid=payment        
        return self.df.loc["Cash Left for Distribution"].loc[i],debt_paid

class Home:
    def __init__(self,ind,spendable,cash,current,amount,monthly_add):
        down_needed=amount-current-monthly_add
        spend_used=min(down_needed,spendable)
        cash_left=min(spendable,spendable-spend_used)
        cash[0]=cash_left
        new_down=current+monthly_add+spend_used
        self.df=create_df(["Down Payment Saved","Monthly Addition","Down Payment Needed","Monthly Spendable Used","New Down Payment Balance","Cash Left for Distribution"],ind,[current,monthly_add,down_needed,spend_used,new_down,cash_left])  
        
    def update(self,i,spendable,amount,monthly_add):
        self.df.loc["Down Payment Saved"].loc[i]=self.df.loc["New Down Payment Balance"].loc[i-1]
        self.df.loc["Monthly Addition"].loc[i]=monthly_add
        self.df.loc["Down Payment Needed"].loc[i]=amount-self.df.loc["Monthly Addition"].loc[i]-self.df.loc["Down Payment Saved"].loc[i]
        self.df.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.df.loc["Down Payment Needed"].loc[i])
        self.df.loc["New Down Payment Balance"].loc[i]=self.df.loc["Down Payment Saved"].loc[i]+self.df.loc["Monthly Addition"].loc[i]+self.df.loc["Monthly Spendable Used"].loc[i]
        self.df.loc["Cash Left for Distribution"].loc[i]=spendable-self.df.loc["Monthly Spendable Used"].loc[i]
        return self.df.loc["Cash Left for Distribution"].loc[i]
        
class IRA:
    def __init__(self,ind,spendable,cash,max_cont,cur_cont):
        spend_used=min(max_cont-cur_cont,spendable)
        new_cont=cur_cont+spend_used
        cash_left=spendable-spend_used
        cash[0]=cash_left
        self.df=create_df(["Maximum Monthly","Current Contribution","Monthly Spendable Used","New Monthly Contribution","Cash Left for Distribution"],ind,[max_cont,cur_cont,spend_used,new_cont,cash_left])  
        
    def update(self,i,spendable,max_cont):
        self.df.loc["Maximum Monthly"].loc[i]=max_cont
        self.df.loc["Current Contribution"].loc[i]=self.df.loc["New Monthly Contribution"].loc[i-1]
        self.df.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.df.loc["Maximum Monthly"].loc[i]-self.df.loc["Current Contribution"].loc[i])
        self.df.loc["New Monthly Contribution"].loc[i]=self.df.loc["Current Contribution"].loc[i]+self.df.loc["Monthly Spendable Used"].loc[i]
        self.df.loc["Cash Left for Distribution"].loc[i]=spendable-self.df.loc["Monthly Spendable Used"].loc[i]
        return self.df.loc["Cash Left for Distribution"].loc[i]
    
class Retirement:
    def __init__(self,ind,spendable,cash,cur_savings,other_savings,rate,needed_savings,cont_ira,cont_401k,months_left):
        future_val=cur_savings*(1+rate/12)**months_left
        future_val_monthly=((other_savings+cont_ira+cont_401k)*((((1 + (rate / 12)) ** months_left) - 1) / (rate/ 12)))
        spend_needed=needed_savings-cont_401k-other_savings
        spend_used=min(spend_needed,spendable)
        ending_bal=(cur_savings*(1+(rate/12))) + (other_savings+cont_ira+cont_401k+spend_used)
        cash_left=spendable-spend_used
        cash[0]=cash_left
        retirement_val=future_val+(other_savings+cont_ira+cont_401k+spend_used) * ((((1 + (rate/12)) ** months_left) - 1) / (rate/12))
        self.df=create_df(["Current Savings","Other Monthly Contribution","IRA Monthly Contribution","401K Monthly Contribution","Future Value of Current Savings","Future Value of Monthly Savings","Monthly Needed","Monthly Spendable Used","Ending Balance","Cash Left for Distribution","Future Value At Retirement"],ind,[cur_savings,other_savings,cont_ira,cont_401k,future_val,future_val_monthly,spend_needed,spend_used,ending_bal,cash_left,retirement_val]) 
        
    def update(self,i,spendable,rate,needed_savings,cont_ira,cont_401k,months_left):
        self.df.loc["Current Savings"].loc[i]=self.df.loc["Ending Balance"].loc[i-1]
        self.df.loc["Other Monthly Contribution"].loc[i]=self.df.loc["Other Monthly Contribution"].loc[i-1]+self.df.loc["Monthly Spendable Used"].loc[i-1]
        self.df.loc["IRA Monthly Contribution"].loc[i]=cont_ira
        self.df.loc["401K Monthly Contribution"].loc[i]=cont_401k
        self.df.loc["Future Value of Current Savings"].loc[i]=self.df.loc["Current Savings"].loc[i]*((1+(rate/12))**months_left)
        self.df.loc["Future Value of Monthly Savings"].loc[i]=((self.df.loc["Other Monthly Contribution"].loc[i]+self.df.loc["IRA Monthly Contribution"].loc[i]+self.df.loc["401K Monthly Contribution"].loc[i]) * ((((1 + (rate / 12))**months_left) - 1) / (rate/12)))
        self.df.loc["Monthly Needed"].loc[i]=max(0, (needed_savings-(self.df.loc["Future Value of Current Savings"].loc[i]+self.df.loc["Future Value of Monthly Savings"].loc[i])) / ((((1 + (rate/12)) ** months_left) - 1) / (rate / 12)))
        self.df.loc["Monthly Spendable Used"].loc[i]=min(spendable,self.df.loc["Monthly Needed"].loc[i])
        self.df.loc["Ending Balance"].loc[i]=(self.df.loc["Current Savings"].loc[i]*(1+(rate/12))) + (self.df.loc["Other Monthly Contribution"].loc[i]+self.df.loc["IRA Monthly Contribution"].loc[i]+self.df.loc["401K Monthly Contribution"].loc[i]+self.df.loc["Monthly Spendable Used"].loc[i])
        self.df.loc["Cash Left for Distribution"].loc[i]=spendable-self.df.loc["Monthly Spendable Used"].loc[i]
        self.df.loc["Future Value At Retirement"].loc[i]=self.df.loc["Future Value of Current Savings"].loc[i]+(self.df.loc["Other Monthly Contribution"].loc[i]+self.df.loc["IRA Monthly Contribution"].loc[i]+self.df.loc["401K Monthly Contribution"].loc[i]+self.df.loc["Monthly Spendable Used"].loc[i]) * ((((1 + (rate/12))**months_left) - 1) / (rate/12))
        return self.df.loc["Cash Left for Distribution"].loc[i]

#Main class
class Pay_opt:
	def __init__(self, Salary, Age_Now, Inflation_Rate, Monthly_Income, Monthly_Expenses, Time_Frame_Desired, Mortgage_Down_Payment_Amt, Mortgage_Down_Payment_Savings_Current, Mortgage_Down_Payment_Savings_Monthly, Rate_of_Return, t401K_Tiers, t401K_Tier1_Match_Pct, t401K_Tier1_Up_To_Pct, t401K_Tier2_Match_Pct, t401K_Tier2_Up_To_Pct, t401K_Deferral_Max_Pct, t401K_Deferral_Min_Pct, t401K_Contribution_Current_Pct, t401K_Company_Match_Max_Pct, Years_Until_Retirement, Retirement_Savings_Needed, Monthly_Retirement_Savings_Needed, Current_Retirement_Savings, Other_Retirement_Savings_Monthly, Debt_Credit_Card_Balance, Debt_Credit_Card_Interest_APR, Debt_Credit_Card_Payment_Monthly, Debt_Student_Loan_Balance, Debt_Student_Loan_Interest_APR, Debt_Student_Loan_Payment_Monthly, Debt_Student_Loan_Payment_Additional, Emergency_Fund_Months_Needed, Emergency_Fund_Savings_Current, Monthly_IRA_Contribution_Max, Monthly_IRA_Contribution_Current,goals):
		self.Salary=Salary
		self.Age_Now=Age_Now
		self.Inflation_Rate=Inflation_Rate
		self.Monthly_Income=Monthly_Income
		self.Monthly_Expenses=Monthly_Expenses
		self.Time_Frame_Desired=Time_Frame_Desired
		self.Mortgage_Down_Payment_Amt=Mortgage_Down_Payment_Amt
		self.Mortgage_Down_Payment_Savings_Current=Mortgage_Down_Payment_Savings_Current
		self.Mortgage_Down_Payment_Savings_Monthly=Mortgage_Down_Payment_Savings_Monthly
		self.Rate_of_Return=Rate_of_Return
		self.t401K_Tiers=t401K_Tiers
		self.t401K_Tier1_Match_Pct=t401K_Tier1_Match_Pct
		self.t401K_Tier1_Up_To_Pct=t401K_Tier1_Up_To_Pct
		self.t401K_Tier2_Match_Pct=t401K_Tier2_Match_Pct
		self.t401K_Tier2_Up_To_Pct=t401K_Tier2_Up_To_Pct
		self.t401K_Deferral_Max_Pct=t401K_Deferral_Max_Pct
		self.t401K_Deferral_Min_Pct=t401K_Deferral_Min_Pct
		self.t401K_Contribution_Current_Pct=t401K_Contribution_Current_Pct
		self.t401K_Company_Match_Max_Pct=t401K_Company_Match_Max_Pct
		self.Years_Until_Retirement=Years_Until_Retirement
		self.Retirement_Savings_Needed=Retirement_Savings_Needed
		self.Monthly_Retirement_Savings_Needed=Monthly_Retirement_Savings_Needed
		self.Current_Retirement_Savings=Current_Retirement_Savings
		self.Other_Retirement_Savings_Monthly=Other_Retirement_Savings_Monthly
		self.Debt_Credit_Card_Balance=Debt_Credit_Card_Balance
		self.Debt_Credit_Card_Interest_APR=Debt_Credit_Card_Interest_APR
		self.Debt_Credit_Card_Payment_Monthly=Debt_Credit_Card_Payment_Monthly
		self.Debt_Student_Loan_Balance=Debt_Student_Loan_Balance
		self.Debt_Student_Loan_Interest_APR=Debt_Student_Loan_Interest_APR
		self.Debt_Student_Loan_Payment_Monthly=Debt_Student_Loan_Payment_Monthly
		self.Debt_Student_Loan_Payment_Additional=Debt_Student_Loan_Payment_Additional
		self.Emergency_Fund_Months_Needed=Emergency_Fund_Months_Needed
		self.Emergency_Fund_Savings_Current=Emergency_Fund_Savings_Current
		self.Monthly_IRA_Contribution_Max=Monthly_IRA_Contribution_Max
		self.Monthly_IRA_Contribution_Current=Monthly_IRA_Contribution_Current
		self.goals=goals		
	
	#initalize goals; note that constructors have no return value, so instead we pass the mutable list
	#cash which has its value set to the cash left for distribution.  Each function is also passed 
	#cash[0] which gives the remaining spendable	
	def init_goals(self):
		self.ind=list(range(self.Years_Until_Retirement*12))
		
		#Goals should be allowed
		master_goals=["Emergency 1","401K Match","Credit Card Debt","401K Deferral","Emergency 2","Home","IRA","Retirement","Student Loan"]
		if not set(self.goals).issubset(set(master_goals)):
			raise NotImplementedError("Only use implemented goals")
		
		#Goals should be unique
		if len(set(self.goals))!=len(self.goals):
			raise NameError("Goals need to be unique")
		
		#Make sure that all the retirement goals are defined sensibly
		if ("401K Match" in self.goals) and ("401K Deferral" not in self.goals):
			raise NotImplementedError("401K Deferral not implemented")
		elif ("401K Match" not in self.goals) and ("401K Deferral" in self.goals):
			raise NotImplementedError("401K Match not implemented")
		elif ("401K Match" not in self.goals) and ("401K Deferral" not in self.goals):
			self.i_401k=c_401k(self.ind,0,[0],0,0,0)
			self.i_401k.deferral_init(self.ind,0,[0],1,0)
		elif self.goals.index("401K Match")>self.goals.index("401K Deferral"):
			raise NotImplementedError("401K Match needs to be implemented before Deferral")
		
		if "IRA" not in self.goals:
			self.ira=IRA(self.ind,0,[0],0,0)
			
		if "Retirement" in self.goals:
			if "401K Deferral" in self.goals and self.goals.index("401K Deferral")>self.goals.index("Retirement"):
				raise NotImplementedError("If 401K is included, it needs to be implemented before retirement goal")
			if "IRA" in self.goals and self.goals.index("IRA")>self.goals.index("Retirement"):
				raise NotImplementedError("If IRA is included, it needs to be implemented before retirement goal")
		else:
			self.retirement=Retirement(self.ind,0,[0],0,0,1,0,0,0,0)
		
		#Make sure that emergency fund is defined sensibly
		if ("Emergency 1" in self.goals) and ("Emergency 2" not in self.goals):
			raise NotImplementedError("Emergency 2 not implemented")
		if ("Emergency 1" not in self.goals) and ("Emergency 2" in self.goals):
			raise NotImplementedError("Emergency1 not implemented")		
		if ("Emergency 1" in self.goals) and ("Emergency 2" in self.goals) and (self.goals.index("Emergency 1")>self.goals.index("Emergency 2")):
			raise NotImplementedError("Emergency 1 needs to be implemented before Emergency2")
					
		#initialize main data frame
		self.d=create_df(["Salary","Month","Months Until Retirement","Income","Completed Debt Payments","Reallocated Funds Home","Reallocated Funds 2", "Expenses","Allocated To 401K","Allocated to IRA & Other","Monthly Spendable"],self.ind,[self.Salary,1,self.Years_Until_Retirement*12,self.Monthly_Income,0,self.Mortgage_Down_Payment_Savings_Monthly, self.Debt_Student_Loan_Payment_Additional, self.Monthly_Expenses-self.Mortgage_Down_Payment_Savings_Monthly-self.Debt_Student_Loan_Payment_Additional,0,0,self.Monthly_Income-(self.Monthly_Expenses-self.Mortgage_Down_Payment_Savings_Monthly-self.Debt_Student_Loan_Payment_Additional) ])
		cash=[self.d.loc["Monthly Spendable"].loc[0] ]
		
		#initialize goals
		for g in self.goals:
			if g=="Emergency 1":		
				self.emergency=Emergency(self.ind,cash[0],cash,self.d.loc["Expenses"].loc[0],self.Emergency_Fund_Savings_Current)
			elif g=="401K Match":
				self.i_401k=c_401k(self.ind,cash[0],cash,self.t401K_Contribution_Current_Pct,self.t401K_Deferral_Min_Pct,self.Salary)
			elif g=="Credit Card Debt":
				self.debt_cc=Debt(self.ind,cash[0],cash,self.Debt_Credit_Card_Balance,self.Debt_Credit_Card_Interest_APR,self.Debt_Credit_Card_Payment_Monthly)
			elif g=="401K Deferral":
				self.i_401k.deferral_init(self.ind,cash[0],cash,self.d.loc["Salary"].loc[0],self.t401K_Deferral_Max_Pct)
				self.i_401k.stats_init(self.ind,self.d.loc["Salary"].loc[0],self.t401K_Tier1_Match_Pct,self.t401K_Tier1_Up_To_Pct,self.t401K_Tier2_Match_Pct,self.t401K_Tier2_Up_To_Pct,self.t401K_Company_Match_Max_Pct)
			elif g=="Emergency 2":
				self.emergency.fund2_init(self.ind,cash[0],cash,self.d.loc["Expenses"].loc[0],self.Emergency_Fund_Months_Needed)
			elif g=="Home":
				self.home=Home(self.ind,cash[0],cash,self.Mortgage_Down_Payment_Savings_Current,self.Mortgage_Down_Payment_Amt,0)
			elif g=="IRA":
				self.ira=IRA(self.ind,cash[0],cash,self.Monthly_IRA_Contribution_Max,self.Monthly_IRA_Contribution_Current)
			elif g=="Retirement":
				self.retirement=Retirement(self.ind,cash[0],cash,self.Current_Retirement_Savings,self.Other_Retirement_Savings_Monthly, self.Rate_of_Return, self.Monthly_Retirement_Savings_Needed, self.ira.df.loc["New Monthly Contribution"].loc[0], self.i_401k.ddeferral.loc["Current Monthly Contribution"].loc[0]+self.i_401k.ddeferral.loc["Monthly Spendable Used"].loc[0]+self.i_401k.stats.loc["Match $"].loc[0], self.d.loc["Months Until Retirement"].loc[0])
			elif g=="Student Loan":
				self.stud_loan=Debt(self.ind,cash[0],cash,self.Debt_Student_Loan_Balance,self.Debt_Student_Loan_Interest_APR,self.Debt_Student_Loan_Payment_Monthly)
			else:
				raise NotImplementedError("Only use implemented goals")
	#update goals
	def update(self):
		cc_paid=0
		stud_loan_paid=0
		
		init_dict={}
		init_dict["Emergency 1"]=lambda i,cash: self.emergency.fund1_update(i,cash)
		init_dict["401K Match"]=lambda i,cash: self.i_401k.match_update(i,cash,self.t401K_Deferral_Min_Pct,self.d.loc["Salary"].loc[i-1])
		init_dict["Credit Card Debt"]=lambda i,cash: self.debt_cc.update(i,cash,self.Debt_Credit_Card_Interest_APR,self.Debt_Credit_Card_Payment_Monthly)
		init_dict["401K Deferral"]=lambda i,cash: (self.i_401k.deferral_update(i,cash,self.d.loc["Salary"].loc[i],self.t401K_Deferral_Max_Pct), self.i_401k.stats_update(i,self.d.loc["Salary"].loc[i],self.t401K_Tier1_Match_Pct,self.t401K_Tier1_Up_To_Pct,self.t401K_Tier2_Match_Pct,self.t401K_Tier2_Up_To_Pct,self.t401K_Company_Match_Max_Pct))[0]
		init_dict["Emergency 2"]=lambda i,cash: self.emergency.fund2_update(i,cash)
		init_dict["Home"]=lambda i,cash: self.home.update(i,cash,self.Mortgage_Down_Payment_Amt,0)
		init_dict["IRA"]=lambda i,cash: self.ira.update(i,cash,self.Monthly_IRA_Contribution_Max)
		init_dict["Retirement"]=lambda i,cash: self.retirement.update(i,cash,self.Rate_of_Return,self.Retirement_Savings_Needed,self.ira.df.loc["New Monthly Contribution"].loc[i],self.i_401k.ddeferral.loc["Current Monthly Contribution"].loc[i]+self.i_401k.ddeferral.loc["Monthly Spendable Used"].loc[i]+self.i_401k.stats.loc["Match $"].loc[i],self.d.loc["Months Until Retirement"].loc[i])
		init_dict["Student Loan"]=lambda i,cash: self.stud_loan.update(i,cash,self.Debt_Student_Loan_Interest_APR,self.Debt_Student_Loan_Payment_Monthly)
				

		for i in self.ind[1:]:
		    self.d.loc["Salary"].loc[i]=self.Salary*(1+self.Inflation_Rate)**(int(i/12))
		    self.d.loc["Month"].loc[i]=1+i
		    self.d.loc["Months Until Retirement"].loc[i]=12*self.Years_Until_Retirement-i
		    self.d.loc["Income"].loc[i]=self.Monthly_Income*(1+self.Inflation_Rate)**(int(i/12))
		    self.d.loc["Completed Debt Payments"].loc[i]=cc_paid+stud_loan_paid
		    self.d.loc["Reallocated Funds Home"].loc[i]=self.Mortgage_Down_Payment_Savings_Monthly
		    self.d.loc["Reallocated Funds 2"].loc[i]=self.Debt_Student_Loan_Payment_Additional
		    self.d.loc["Expenses"].loc[i]=(self.d.loc["Expenses"].loc[i-1]-self.d.loc["Completed Debt Payments"].loc[i])*(1+self.Inflation_Rate)**(int((i)/12)>int((i-1)/12))
		    self.d.loc["Allocated To 401K"].loc[i]=(self.d.loc["Allocated To 401K"].loc[i-1]+self.i_401k.dmatch.loc["Monthly Spendable Used"].loc[i-1]+self.i_401k.ddeferral.loc["Monthly Spendable Used"].loc[i-1])*(1+self.Inflation_Rate)**(int((i)/12)>int((i-1)/12))
		    self.d.loc["Allocated to IRA & Other"].loc[i]=self.d.loc["Allocated to IRA & Other"].loc[i-1]+self.ira.df.loc["Monthly Spendable Used"].loc[i-1]+self.retirement.df.loc["Monthly Spendable Used"].loc[i-1]
		    self.d.loc["Monthly Spendable"].loc[i]=self.d.loc["Income"].loc[i]-self.d.loc["Expenses"].loc[i]-self.d.loc["Allocated To 401K"].loc[i]-self.d.loc["Allocated to IRA & Other"].loc[i]
		    
		    #Cash is no longer a mutable list, since we can just use return values from now on
		    cash=self.d.loc["Monthly Spendable"].loc[i]
		    for g in self.goals:
		    	#need special checks for debt, since it returns two values instead of one
		    	if g=="Credit Card Debt":
		    		cash,cc_paid=init_dict[g](i,cash)
		    	elif g=="Student Loan":
		    		cash,stud_loan_paid=init_dict[g](i,cash)	    
		    	else:
		    		cash=init_dict[g](i,cash)
		    
	
	def print(self):
		print(self.d)
		for g in self.goals:
			print(g+"\n")
			if g=="Emergency 1":		
				print(self.emergency.df1)
			elif g=="401K Match":
				print(self.i_401k.dmatch)
			elif g=="Credit Card Debt":
				print(self.debt_cc.df)
			elif g=="401K Deferral":
				print(self.i_401k.ddeferral)
				print(self.i_401k.stats)
			elif g=="Emergency 2":
				print(self.emergency.df2)
			elif g=="Home":
				print(self.home.df)
			elif g=="IRA":
				print(self.ira.df)
			elif g=="Retirement":
				print(self.retirement.df)
			elif g=="Student Loan":
				print(self.stud_loan.df)
			else:
				raise NotImplementedError("Only use implemented goals")		
	
	def results(self,goals):
		self.goals=goals
		self.init_goals()
		self.update()
	
	
goal=["Emergency 1","401K Match","Credit Card Debt","401K Deferral","Emergency 2","Home","IRA","Retirement","Student Loan"]
pay_opt=Pay_opt(Salary, Age_Now, Inflation_Rate, Monthly_Income, Monthly_Expenses, Time_Frame_Desired, Mortgage_Down_Payment_Amt, Mortgage_Down_Payment_Savings_Current, Mortgage_Down_Payment_Savings_Monthly, Rate_of_Return, t401K_Tiers, t401K_Tier1_Match_Pct, t401K_Tier1_Up_To_Pct, t401K_Tier2_Match_Pct, t401K_Tier2_Up_To_Pct, t401K_Deferral_Max_Pct, t401K_Deferral_Min_Pct, t401K_Contribution_Current_Pct, t401K_Company_Match_Max_Pct, Years_Until_Retirement, Retirement_Savings_Needed, Monthly_Retirement_Savings_Needed, Current_Retirement_Savings, Other_Retirement_Savings_Monthly, Debt_Credit_Card_Balance, Debt_Credit_Card_Interest_APR, Debt_Credit_Card_Payment_Monthly, Debt_Student_Loan_Balance, Debt_Student_Loan_Interest_APR, Debt_Student_Loan_Payment_Monthly, Debt_Student_Loan_Payment_Additional, Emergency_Fund_Months_Needed, Emergency_Fund_Savings_Current, Monthly_IRA_Contribution_Max, Monthly_IRA_Contribution_Current,goal)

pay_opt.results(goal)
pay_opt.print()

goal=["401K Match","Credit Card Debt","401K Deferral","Home","IRA","Retirement","Student Loan"]
pay_opt.results(goal)
pay_opt.print()

