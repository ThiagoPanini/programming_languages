package com.projeto.projetoprogweb.rss;

import java.net.MalformedURLException;
import java.net.URL;
import java.io.*;


public class ReadRSS {

    public static void main(String[] args) {

        System.out.println(readRSSFeed("http://g1.globo.com/dynamo/brasil/rss2.xml"));
    }

    public static String readRSSFeed(String urlAddress){
        try{
            URL rssUrl = new URL (urlAddress);
            BufferedReader in = new BufferedReader(new InputStreamReader(rssUrl.openStream(),"UTF-8"));
            String sourceCode = "";
            String line;
            while((line=in.readLine())!=null){
                if(line.contains("<title>")){
                    System.out.println(line);
                    int firstPos = line.indexOf("<title>");
                    String temp = line.substring(firstPos);
                    temp=temp.replace("<title>","");
                    int lastPos = temp.indexOf("</title>");
                    temp = temp.substring(0,lastPos);
                    sourceCode +=temp+ "\n" ;
                }
            }
            in.close();
            return sourceCode;
        } catch (MalformedURLException ue){
            System.out.println("Malformed URL");
        } catch (IOException ioe){
            System.out.println("Something went wrong reading the contents");
        }
        return null;
    }
}
