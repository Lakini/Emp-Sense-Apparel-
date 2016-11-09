/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prediction.apparel.dbtocsv;

import java.io.FileWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import prediction.apparel.databaseConn.DBAccessJava;

/**
 *
 * @author Lakini
 */
public class DbToCSV {
    public void dbToClassConerter(String fileName,String query)
    {
        String filename =fileName;
        try {
            FileWriter fw = new FileWriter(filename);
            DBAccessJava dbaccess = DBAccessJava.getDbCon();
            ResultSet rs= dbaccess.query(query);
            while (rs.next()) {
                
                ResultSetMetaData rsmd = rs.getMetaData();
                int columnsNumber = rsmd.getColumnCount();
                for(int i=0;i<columnsNumber;i++){
                    fw.append(rs.getString(i));
                    if(i<columnsNumber-1)
                        fw.append(',');
                    else
                        fw.append('\n');
                }
               }
            fw.flush();
            fw.close();
            dbaccess.closeConn();
            System.out.println("CSV File is created successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    ////////////////////////////
    
}
