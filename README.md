# StormIncidence
Have the frequency and severity of storms increased since the good old days?

It seems as if the number of very notable storms has increased in recent decades.  The disasters seem to be more frequent, harder hitting with a greater cost in property and lives.  Is this true or is recency bias sneaking up on me?  How many storms can I expect to see in the next five years?  If there really are more storms, when did that change occur.

Scientific restatement of the question.  Reference Data set source.  Provide specific goal such as a predictive model, hypothesis testing.  

*Source Data*
I downloaded storm data going back to 1950 from the National Atmospheric and Oceanic Organization.  It covers intensity of wind, property and crop damage estimates, as well as measures off deaths and injuries related to storms.  
<https://www.ncdc.noaa.gov/stormevents/ftp.jsp>

*Confouding factors*
It appears that the collection of data at NOAA changed artound 1993.  The number of observations increased dramatically.  However, it appears that the most significan t change was an increase in the recording of less dramatic storms.  While the details of these storms may be interesting, they are not poignant at this point.  I focused on higher severity storms.  Go into cutoff criteria, fitting damage data to GDP basis.

*Test*
I will test whether the number of severe storms over the second half of the data set, from 1984 to 2017, is significantly different than the prior period from 1950 to 1983.  I will test across 3 metrics to capture slightly different aspects of of the general question.  First, I will measure the number of storms with damage above a specific criteria adjusted by by real economic growth and inflation.  Second, the number of storms with winds speeds in the highest quartile of intensity.  Third, I will measure aggregate deaths both adjusted and unadjusted by population.  ... if I have time.

*H0*
My null hypothesis is that the number of severe storms in the earlier and later periods come from the same population and that there is no statistical difference between the two epochs.

*H1*
My alternate hypothesis is that there is a difference in the number of severe storms between the two periods.  

*Test Criteria*
My intuition is that storm incidence has increased, but the two sided test is more rigorous and given the confounding factors in this analysis, rigor over the test is important.  I will run a binomial test to determine if the counts over the time periods are likely to have come from the same population with a confidence level of 95%.  Thus, if the generated p-value is below 2.5%, I will reject the null hypothesis.
